# SCRIPT TO RUN THE NVT TO EXPLORE THE MOMENTUM- AND POSITION-D.O.F.
# IT IS ASSUMED THAT 1 OR MORE NPT INSTANCES HAVE BEEN DONE AND THAT THEIR mlmd.traj ARE STORED INSIDE NPT/#_instance/NPT/ 
# THE TXXXK_unitcell.json FILE MUST BE PRESENT INSIDE A FOLDER CALLED "NPT", WHICH IS USUALLY CREATED DURING THE NPT PHASE

import numpy as np
import sys
sys.path.append('/scratch/ulg/matnan/slongo/my_scripts/')
from utils_md import run_md
from utils import path
import os
from ase.io import read, write
from ase.build import make_supercell
import pymatgen
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pathlib import Path


# MPIRUN
mpirun = 
# CELL REFINEMENT BOOLEAN
refine =

# PARAMETERS TO CHANGE FOR THE MD
dt =
nsteps =
loginterval = 
nthrow = 
nproc =
lmp_bin = 
pair_style = 
temperature = int(os.getcwd().split('/')[-3][1:-1])
root_dir = 
temp_dir = path(f'{rootdir}T{temperature}K')
mult_mat = 
ismpi =

# As a first thing, we need a copy of the unit cell
ucell_path = f'{rootdir}unit_cell.poscar'
ucell = read(ucell_path)

# We prepare the inverted matrix for the contraction
invmult = np.linalg.inv(mult_mat)

# We look for the npt instances
npt_dir = f'{temp_dir}/NPT/'
npt_inst_dirs = Path(npt_dir).glob('*_instance')

# Now we load the NPT cells (to average)
npt_cells = []
for npt_inst_dir in npt_inst_dirs:
    traj_path = f'{path(npt_inst_dir)}NPT/mlmd.traj'
    traj = read(traj_path, index=':')
    npt_cells.extend([x.get_cell() for x in traj])
npt_cells = np.array(npt_cells)

# Now we need to average the cells
avg_npt_cell = npt_cells.mean(axis=0)

# Now we need to contract the average cell
avg_npt_cell = invmult @ avg_npt_cell

# Now we want to create a unitcell with the new contracted and averaged cell
new_cell = ucell.copy()
new_cell.set_cell(avg_npt_cell, True)

if refine == True:
    # We symmetrize it
    adapter = AseAtomsAdaptor()
    analyzer = SpacegroupAnalyzer(adapter.get_structure(new_cell))
    new_cell = adapter.get_atoms(analyzer.get_refined_structure())

# We also save it
avg_cell_path = f'{temp_dir}T{temperature}K_unitcell.json'
write(avg_cell_path, new_cell)


# Now the actual NVT part
nvt_scell = make_supercell(new_cell, mult_mat)

run_args = dict(ismpi = ismpi,
                mpirun = mpirun,
                temperature = temperature,
                dt = dt,
                nsteps = nsteps,
                loginterval = loginterval,        
                nthrow = nthrow,
                nproc = nproc,
                ucell_path = f'{rootdir}unit_cell.poscar',
                mult_mat = mult_mat,
                pair_style = pair_style,
                pair_coeff = ['* *'],
                lmp_bin = lmp_bin, 
                ase_lammps_command=None,
                wdir = root_dir,
                make_wdir = True,
                NPT = False,
                NVT = True,
                nvt_scell = nvt_scell,
                logfile = 'mlmd.log',
                trajfile = 'mlmd.traj')

run_md(**run_args)


