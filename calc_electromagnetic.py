from __future__ import division
import numpy as np
import interactionRate
import os
import gitHelp as gh
from calc_all import fields_cmbebl, fields_urb
from units import eV, mass_electron, c_light, h_planck, k_boltzmann, sigma_thomson, alpha_finestructure, Mpc

me2 = (mass_electron*c_light**2.) ** 2  # squared electron mass [J^2/c^4]
ENERGY_LOG10_MIN = 0
ENERGY_LOG10_MAX = 23
ENERGY_LOG10_STEP = 0.05
S_KIN_LOG10_MAX = 23
S_KIN_DEFAULT_LOG10_MIN = 4
S_KIN_ICS_LOG10_MIN = -12
S_KIN_RATE_POINTS = 2 ** 18 + 1
S_KIN_CDF_POINTS_PER_DECADE = 2000
S_KIN_SAVE_LOG10_STEP = 0.1
RATE_CHUNK_SIZE = 16
CDF_CHUNK_SIZE = 8
ICS_THOMSON_TRANSITION = 1e8 * eV
PHOTON_DENSITY_POINTS = 200000

def sigmaPP(s):
    """ Pair production cross section (Breit-Wheeler), see Lee 1996 """
    smin = 4 * me2
    if (s < smin):
        return 0.

    b = np.sqrt(1 - smin / s)
    return sigma_thomson * 3 / 16 * (1 - b**2) * ((3 - b**4) * (np.log1p(b) - np.log1p(-b)) - 2 * b * (2 - b**2))


def sigmaDPP(s):
    """ Double-pair production cross section, see R.W. Brown eq. (4.5) with k^2 = q^2 = 0 """
    smin = 16 * me2
    if (s < smin):
        return 0

    return 6.45E-34 * (1 - smin / s)**6


def sigmaICS(s):
    """Inverse Compton scattering cross section.

    The Klein-Nishina expression is used on one continuous s-grid. Very close
    to threshold the analytic Thomson limit avoids cancellation without
    switching to a separate low-energy rate table.
    """
    smin = me2

    if s <= smin:
        return 0.

    delta = s - smin

    # Avoid numerical instability very close to threshold.
    if delta / smin <= 1e-5:
        return sigma_thomson

    b = delta / (s + smin)
    A = 2 / b / (1 + b) * (2 + 2 * b - b**2 - 2 * b**3)
    B = (2 - 3 * b**2 - b**3) / b**2 * (np.log1p(b) - np.log1p(-b))

    return sigma_thomson * 3 / 8 * smin / s / b * (A - B)

def sigmaTPP(s):
    """ Triplet-pair production cross section, see Lee 1996 """
    beta = 28 / 9 * np.log(s / me2) - 218 / 27
    if beta < 0:
        return 0
    
    return sigma_thomson * 3 / 8 / np.pi * alpha_finestructure * beta


def getTabulatedXS(sigma, skin):
    """ Get crosssection for tabulated s_kin """
    if sigma in (sigmaPP, sigmaDPP):  # photon interactions
        return np.array([sigma(s) for s in skin])
    if sigma in (sigmaTPP, sigmaICS):  # electron interactions
        return np.array([sigma(s) for s in skin + me2])
    return False


def getSmin(sigma):
    """ Return minimum required s_kin = s - (mc^2)^2 for interaction """
    return {sigmaPP: 4 * me2,
            sigmaDPP: 16 * me2,
            sigmaTPP: np.exp((218 / 27) / (28 / 9)) * me2 - me2,
            sigmaICS: 1e-40 * me2
            }[sigma]


def getEmin(sigma, field):
    """ Return minimum required cosmic ray energy for interaction *sigma* with *field* """
    return getSmin(sigma) / 4 / field.getEmax()


def photonNumberDensity(field):
    """Return the total photon number density of a background field [1/m^3]."""
    if hasattr(field, "T_CMB"):
        zeta3 = 1.202056903159594
        return (
            16.0
            * np.pi
            * zeta3
            * (k_boltzmann * field.T_CMB) ** 3
            / (h_planck * c_light) ** 3
        )

    eps = np.logspace(
        np.log10(field.getEmin()),
        np.log10(field.getEmax()),
        PHOTON_DENSITY_POINTS
    )
    density = np.asarray(field.getDensity(eps), dtype=float).squeeze()
    density = np.where(np.isfinite(density), density, 0.)
    return np.trapz(density, eps)


def thomsonRate(field):
    """Return the low-energy inverse Compton rate in the Thomson limit [1/Mpc]."""
    return photonNumberDensity(field) * sigma_thomson * Mpc


def getPrimaryEnergyGrid():
    """Return the tabulated primary kinetic-energy grid [J]."""
    n = int(round((ENERGY_LOG10_MAX - ENERGY_LOG10_MIN) / ENERGY_LOG10_STEP)) + 1
    return np.logspace(ENERGY_LOG10_MIN, ENERGY_LOG10_MAX, n) * eV


def getSKinLog10Min(sigma):
    """Return the lower s_kin grid edge in log10(eV^2)."""
    if sigma is sigmaICS:
        return S_KIN_ICS_LOG10_MIN
    return S_KIN_DEFAULT_LOG10_MIN


def getRateSKinGrid(sigma):
    """Return the Romberg-compatible s_kin grid for total rates [J^2]."""
    return np.logspace(
        getSKinLog10Min(sigma),
        S_KIN_LOG10_MAX,
        S_KIN_RATE_POINTS
    ) * eV**2


def getCDFSKinGrid(sigma):
    """Return the high-resolution s_kin grid for cumulative rates [J^2]."""
    log_min = getSKinLog10Min(sigma)
    n = int(round((S_KIN_LOG10_MAX - log_min) * S_KIN_CDF_POINTS_PER_DECADE)) + 1
    return np.logspace(log_min, S_KIN_LOG10_MAX, n) * eV**2


def getSavedSKinGrid(sigma):
    """Return the saved s_kin grid. The 0.1 dex spacing matches C++ sampling."""
    log_min = getSKinLog10Min(sigma)
    n = int(round((S_KIN_LOG10_MAX - log_min) / S_KIN_SAVE_LOG10_STEP)) + 1
    return np.logspace(log_min, S_KIN_LOG10_MAX, n) * eV**2


def calcRateSChunked(s_kin, xs, E, field, cdf=False):
    """Calculate rates in energy chunks to keep the 1 eV extension memory-safe."""
    chunk_size = CDF_CHUNK_SIZE if cdf else RATE_CHUNK_SIZE
    density_primary_energy_max = np.max(E)
    chunks = []
    for i in range(0, len(E), chunk_size):
        chunks.append(
            interactionRate.calc_rate_s(
                s_kin,
                xs,
                E[i:i + chunk_size],
                field,
                cdf=cdf,
                density_primary_energy_max=density_primary_energy_max
            )
        )
    return np.concatenate(chunks, axis=0)


def process(sigma, field, name):
    """ 
        calculate the interaction rates for a given process on a given photon field 

        sigma : crossection (function) of the EM-process
        field : photon field as defined in photonField.py
        name  : name of the process which will be calculated. Necessary for the naming of the data folder
    """

    # output folder
    folder = 'data/' + name
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Tabulated primary kinetic energies. Keep the full grid down to 1 eV so
    # low-energy propagation can use the same data files without extrapolation.
    E = getPrimaryEnergyGrid()
    
    # -------------------------------------------
    # calculate interaction rates
    # -------------------------------------------
    # tabulated values of s_kin = s - mc^2
    # Note: integration method (Romberg) requires 2^n + 1 log-spaced tabulation points
    s_kin = getRateSKinGrid(sigma)
    xs = getTabulatedXS(sigma, s_kin)
    rate = calcRateSChunked(s_kin, xs, E, field)
    if sigma is sigmaICS:
        rate[E <= ICS_THOMSON_TRANSITION] = thomsonRate(field)

    # save
    fname = folder + '/rate_%s.txt' % field.name
    data = np.c_[np.log10(E / eV), rate]
    fmt = '%.2f\t%8.7e'
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s interaction rates\nphoton field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), 1/lambda [1/Mpc]" )
    except:
        header = ("%s interaction rates\nphoton field: %s\n"% (name, field.info)
                  +"log10(E/eV), 1/lambda [1/Mpc]")
    np.savetxt(fname, data, fmt=fmt, header=header)

    # -------------------------------------------
    # calculate cumulative differential interaction rates for sampling s values
    # -------------------------------------------
    # find minimum value of s_kin
    skin1 = getSmin(sigma)  # s threshold for interaction
    skin2 = 4 * field.getEmin() * E[0]  # minimum achievable s in collision with background photon (at any tabulated E)
    skin_min = max(skin1, skin2)

    # tabulated values of s_kin = s - mc^2, limit to relevant range
    # Note: use higher resolution and then downsample
    skin = getCDFSKinGrid(sigma)
    skin = skin[skin > skin_min]

    xs = getTabulatedXS(sigma, skin)
    rate = calcRateSChunked(skin, xs, E, field, cdf=True)

    # downsample
    skin_save = getSavedSKinGrid(sigma)
    skin_save = skin_save[skin_save > skin_min]
    rate_save = np.array([np.interp(skin_save, skin, r) for r in rate])

    # save
    data = np.c_[np.log10(E / eV), rate_save]  # prepend log10(E/eV) as first column
    row0 = np.r_[0, np.log10(skin_save / eV**2)][np.newaxis]
    data = np.r_[row0, data]  # prepend log10(s_kin/eV^2) as first row

    fname = folder + '/cdf_%s.txt' % field.name
    fmt = '%.2f' + '\t%6.5e' * np.shape(rate_save)[1]
    try:
        git_hash = gh.get_git_revision_hash()
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"Produced with crpropa-data version: "+git_hash+"\n"
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row" )
    except:
        header = ("%s cumulative differential rate\nphoton field: %s\n"% (name, field.info)
                  +"log10(E/eV), d(1/lambda)/ds_kin [1/Mpc/eV^2] for log10(s_kin/eV^2) as given in first row")
    np.savetxt(fname, data, fmt=fmt, header=header)

    del data, rate, skin, skin_save, rate_save

if __name__ == "__main__":

    for field in fields_cmbebl + fields_urb:
        print(field.name)
        process(sigmaPP, field, 'EMPairProduction')
        process(sigmaDPP, field, 'EMDoublePairProduction')
        process(sigmaTPP, field, 'EMTripletPairProduction')
        process(sigmaICS, field, 'EMInverseComptonScattering')
