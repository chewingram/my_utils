import numpy as np
from pathlib import Path
import random

from ase.io import read, write
from ase.build import make_supercell

from my_utils.utils_tdep import extract_ifcs

$wdir$
$tdep_bin_directory$
$nthrow$
$rc2$
$rc3$
$traj_path$
$uc_path$
$mult_mat$
$temperature$
$index$
$ts$
$first_order$
$displ_threshold_firstorder$
$max_iterations_first_order$
$polar$
$loto_filepath$
$stride$
$bin_prefix$

# LINE 29; DON'T WRITE ANYTHING ABOVE IT

wdir = Path(wdir)

uc = read(uc_path)

sc = make_supercell(uc, mult_mat)

traj = read(traj_path, index=f'{nthrow}::{stride}')
random.shuffle(traj)
traj = traj[:index]

# TDEP

extract_ifcs(from_infiles = False,
             infiles_dir = None,
             unitcell = uc,
             supercell = sc,
             sampling = traj,
             timestep = ts,
             dir = wdir.absolute(),
             first_order = first_order,
             displ_threshold_firstorder = displ_threshold_firstorder,
             max_iterations_first_order = max_iterations_first_order,
             rc2 = rc2, 
             rc3 = rc3, 
             polar = polar,
             loto_filepath = loto_filepath, 
             stride = 1, # it's already stridden above 
             temperature = temperature,
             bin_prefix = bin_prefix,
             tdep_bin_directory = tdep_bin_directory)



