# SCRIPT TO RUN THE NPT INSTANCES TO EXPLORE THE CELL D.O.F.

import numpy as np
import sys
sys.path.append('/scratch/ulg/matnan/slongo/my_scripts/')
from utils_md import run_md
from utils import path
import os
from ase.io import read, write
from ase.build import make_supercell
from subprocess import run
from pathlib import Path

ninstances = 10
temperature = int(os.getcwd().split('/')[-1][1:-1])
rootdir = #path("/scratch/ulg/matnan/slongo/Work/ML/MoS2/MTP/MD_bigcell/")
temp_dir = path(f'{rootdir}T{temperature}K')
scripts_dir = path(Path('./').resolve())

for iinst in range(ninstances): 
    inst_dir = path(f'{temp_dir}NPT/{iinst}_instance/')
    os.system(f'mkdir -p {inst_dir}')
    os.system(f'ln -s -f {scripts_dir}RunNPT_instance.py {scripts_dir}npt_job.sh {inst_dir}')
    cmd = f'sbatch npt_job.sh'
    run(cmd.split(), cwd=inst_dir)
    
                  

