# SCRIPT TO RUN THE NPT INSTANCES TO EXPLORE THE CELL D.O.F.

import numpy as np
import sys
sys.path.append('/scratch/ulg/matnan/slongo/my_scripts/')
from my_utils.utils_md import run_md
from my_utils.utils import path
import os
from ase.io import read, write
from ase.build import make_supercell
from subprocess import run
from pathlib import Path

ninstances = 10
temperature = int(os.getcwd().split('/')[-1][1:-1])
root_dir = 
root_dir = Path(root_dir)
temp_dir = Path(root_dir).joinpath(f'T{temperature}K')
scripts_dir = Path('./').resolve()

for iinst in range(ninstances): 
    inst_dir = temp_dir.joinpath(f'NPT/{iinst}_instance/')
    os.system(f'mkdir -p {inst_dir}')
    os.system(f"ln -s -f {temp_dir.joinpath('RunNPT_instance.py').resolve()} {temp_dir.joinpath('npt_job.sh').resolve()} {inst_dir.resolve()}")
    cmd = f'sbatch npt_job.sh'
    run(cmd.split(), cwd=inst_dir)
    
                  

