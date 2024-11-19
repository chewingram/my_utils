# SCRIPT TO RUN THE NVT TO EXPLORE THE MOMENTUM- AND POSITION-D.O.F.
# IT IS ASSUMED THAT 1 OR MORE NPT INSTANCES HAVE BEEN DONE AND THAT THEIR mlmd.traj ARE STORED INSIDE NPT/#_instance/NPT/ 
# THE TXXXK_unitcell.json FILE MUST BE PRESENT INSIDE A FOLDER CALLED "NPT", WHICH IS USUALLY CREATED DURING THE NPT PHASE

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
root_dir = 
mult_mat = 
ismpi = 

import numpy as np
import sys
sys.path.append('/scratch/ulg/matnan/slongo/my_scripts/')
from my_utils.utils_md import run_md
from my_utils.utils import path
import os
from ase.io import read, write
from ase.build import make_supercell
from ase.spacegroup.symmetrize import refine_symmetry
#import pymatgen
#from pymatgen.io.ase import AseAtomsAdaptor
#from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
#from pathlib import Path


temperature = int(os.getcwd().split('/')[-1][1:-1])

root_dir = Path(root_dir)
temp_dir = root_dir.joinpath(f'T{temperature}K')

# As a first thing, we need a copy of the unit cell
ucell_path = root_dir.joinpath(f'unitcell.poscar')
ucell = read(ucell_path)

# We prepare the inverted matrix for the contraction
invmult = np.linalg.inv(mult_mat)

# We look for the npt instances
npt_dir = temp_dir.joinpath(f'NPT/')
npt_inst_dirs = Path(npt_dir).glob('*_instance')
# Now we load the NPT cells (to average)
npt_cells = []
for npt_inst_dir in npt_inst_dirs:
    traj_path = npt_inst_dir.joinpath('NPT/Trajectory/mlmd.traj')
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
    #adapter = AseAtomsAdaptor()
    #analyzer = SpacegroupAnalyzer(adapter.get_structure(new_cell))
    #new_cell = adapter.get_atoms(analyzer.get_refined_structure())
    refine_symmetry(new_cell)

# We also save it
avg_cell_path = temp_dir.joinpath(f'T{temperature}K_unitcell.json')
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
                ucell_path = root_dir.joinpath('unitcell.poscar'),
                mult_mat = mult_mat,
                pair_style = pair_style,
                pair_coeff = ['* *'],
                lmp_bin = lmp_bin, 
                ase_lammps_command=None,
                wdir = Path('./'),
                make_wdir = True,
                NPT = False,
                NVT = True,
                nvt_scell = nvt_scell,
                logfile = 'mlmd.log',
                trajfile = 'mlmd.traj')

run_md(**run_args)


