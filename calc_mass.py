import numpy as np
import gitHelp as gh
import os

from units import amu, mass_electron, mass_proton, mass_neutron, keV, c_squared

cdir = os.path.split(__file__)[0]

# This script generates a table of nuclear mass for all combinations (Z,N) Z=0..82, N=0..132
# Priority order for each isotope:
#   1. NIST measured atomic mass (tables/mass_NIST.txt)
#   2. Mass excess from NuDat3 decay table (tables/decay_NuDat3.txt, column 6, in keV)
#   3. Approximation: A * amu - Z * mass_electron
# The nuclear mass is approximated as atomic mass minus Z * electron mass.
# Electron binding energies (~keV) are negligible compared to nuclear binding energies (~MeV).

ZMAX = 82   # maximum proton number (Pb)
NMAX = 132  # maximum neutron number (Pb-214)

D = np.zeros((ZMAX + 1, NMAX + 1))

### read NIST data
# See: http://www.nist.gov/pml/data/comp.cfm
# All Isotopes, Linearized ASCII Output
# http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii2&isotype=all
datapath = os.path.join(cdir, 'tables/mass_NIST.txt')
fin = open(datapath, 'r')

for i in range(4):
    fin.readline() # skip header

z = 0
a = 0
for line in fin.readlines():
    if line.startswith('Atomic Number'):
        z = int(line.strip('Atomic Number = '))
        continue

    if line.startswith('Mass Number'):
        a = int(line.strip('Mass Number = '))
        continue

    if line.startswith('Relative Atomic Mass'):
        line = line.strip('Relative Atomic Mass = ')
        relAtomicMass = line[:line.find('(')]
        n = a - z
        if a == 1:
            continue # skip H-1
        if z > ZMAX or n > NMAX or n < 0:
            continue
        if not relAtomicMass.strip():
            continue # no measured value

        D[z, n] = float(relAtomicMass) * amu - z * mass_electron

fin.close()

### add neutron and proton mass
D[1, 0] = mass_proton
D[0, 1] = mass_neutron

### fill remaining gaps from NuDat3 mass excess (keV)
# mass_atomic = A * amu + mass_excess * keV / c_squared
# mass_nuclear = mass_atomic - Z * mass_electron
datapath = os.path.join(cdir, 'tables/decay_NuDat3.txt')
with open(datapath) as f:
    lines = f.readlines()

for line in lines[1:]:
    cols = line.split('\t')
    if len(cols) < 7:
        continue
    try:
        Z = int(cols[2].strip())
        N = int(cols[3].strip())
    except ValueError:
        continue
    if Z > ZMAX or N > NMAX:
        continue
    if D[Z, N] != 0:
        continue  # already filled by NIST
    mass_exc_str = cols[6].strip()
    if not mass_exc_str:
        continue  # no mass excess available
    try:
        mass_exc_keV = float(mass_exc_str)
    except ValueError:
        continue
    A = Z + N
    D[Z, N] = A * amu + mass_exc_keV * keV / c_squared - Z * mass_electron

### fill remaining empty entries with A * amu - Z * m_e approximation
for z in range(ZMAX + 1):
    for n in range(NMAX + 1):
        if D[z, n] == 0:
            D[z, n] = (z + n) * amu - z * mass_electron

# output folder
folder = 'data'
if not os.path.exists(folder):
    os.makedirs(folder)
# Write to file
fout = open('data/nuclear_mass.txt', 'w')

# Add git hash of crpropa-data repository to header
try:
    git_hash = gh.get_git_revision_hash()
    fout.write('# Produced with crpropa-data version: '+git_hash+'\n')
except:
    pass

fout.write('# Nuclear mass of isotopes Z=0..%i, N=0..%i\n' % (ZMAX, NMAX))
fout.write('# Sources: NIST atomic masses > NuDat3 mass excess > A*amu approximation\n')
for z in range(ZMAX + 1):
    for n in range(NMAX + 1):
        fout.write(str(z) + ' ' + str(n) + ' ' + str(D[z, n]) + '\n')

fout.close()