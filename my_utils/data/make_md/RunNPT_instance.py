# SCRIPT TO RUN THE NPT TO EXPLORE THE PHACE SPACE ALONG THE CELL-DEGREES OF FREEDOM
# AFTER THIS MULTIPLE INSTANCES OF NVT CAN BE RUN (DISCARDING SOME INITIAL TIMESTEPS) TO EXPLORE THE MOMENTUM- AND POSITION-D.O.F.

# MPIRUN
mpirun =

# PARAMETERS TO CHANGE FOR THE MD
dt =
nsteps =
loginterval =
nthrow =
nproc =
lmp_bin = 
pair_style = 
iso = 
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
from pathlib import Path

temperature = int(os.getcwd().split('/')[-3][1:-1])

root_dir = Path(root_dir)

temp_dir = root_dir.joinpath(f'T{temperature}K')



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
                NPT = True,
                NVT = False,
                nvt_scell = None,
                logfile = 'mlmd.log',
                trajfile = 'mlmd.traj',
                iso = iso)

run_md(**run_args)
                  

