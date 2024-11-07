# SCRIPT TO RUN THE NPT TO EXPLORE THE PHACE SPACE ALONG THE CELL-DEGREES OF FREEDOM
# AFTER THIS MULTIPLE INSTANCES OF NVT CAN BE RUN (DISCARDING SOME INITIAL TIMESTEPS) TO EXPLORE THE MOMENTUM- AND POSITION-D.O.F.

import numpy as np
import sys
sys.path.append('/scratch/ulg/matnan/slongo/my_scripts/')
from utils_md import run_md
from utils import path
import os
from ase.io import read, write


# PARAMETERS TO CHANGE FOR THE MD
dt = 
nsteps = 
loginterval =
nthrow =
nproc = 
iso = 
temperature = int(os.getcwd().split('/')[-3][1:-1])

rootdir = 
temp_dir = path(f'{rootdir}T{temperature}K')

mult_mat =




run_args = dict(ismpi = True,
                mpirun = 'mpirun',
                temperature = temperature,
                dt = dt,
                nsteps = nsteps,
                loginterval = loginterval,        
                nthrow = nthrow,
                nproc = nproc,
                ucell_path = f'{rootdir}unit_cell.poscar',
                mult_mat = mult_mat,
                pair_style = ,# something like 'mlip path_to_the_file/mlip.ini'
                pair_coeff = ['* *'],
                lmp_bin = '/scratch/ulg/matnan/slongo/codes/lammps_mtp/src/lmp_mpi',
                ase_lammps_command=None,
                wdir = path(f'./'),
                make_wdir = True,
                NPT = True,
                NVT = False,
                nvt_scell = None,
                logfile = 'mlmd.log',
                trajfile = 'mlmd.traj',
                iso = iso)

run_md(**run_args)
                  

