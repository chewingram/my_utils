import numpy as np

from pathlib import Path

from ase.io import read, write
from ase.build import make_supercell
from my_utils.utils_tdep import conv_rc2_extract_ifcs, conv_rc3_extract_ifcs


first_order = True
root_dir = Path('./Convergence_tdep')

temperature = int(Path('./').absolute().name[1:-1])

unitcell = read(f'T{temperature}K_unitcell.json')

mult_mat = np.array([[4, 4, 0],
                     [-2, 2, 0],
                     [0, 0, 3]])
supercell = make_supercell(unitcell, mult_mat)

sampling = read(f'T{temperature}K.traj', index=':')

rc2_dir = root_dir.joinpath('rc2')
conv_rc2_extract_ifcs(unitcell = unitcell,
                          supercell = supercell,
                          sampling = sampling,
                          timestep = 1,
                          dir = rc2_dir,
                          first_order = first_order,
                          first_order_rc2 = 20,
                          displ_threshold_firstorder = 0.000001,
                          max_iterations_first_order = 20,
                          rc2s = [4,6,8,10,12,14,16,18,20],  
                          polar = True,
                          loto_filepath = Path('loto_splitting/infile.lotosplitting'), 
                          stride = 1, 
                          temperature = temperature,
                          bin_prefix = 'mpirun',
                          tdep_bin_directory = '',
                          ifc_diff_threshold = 0.000001, # eV/A^2
                          n_rc2_to_average = 4,
                          conv_criterion_diff = 'avg')

if first_order == True:
    unitcell = read(rc2_dir.joinpath('first_order_optimisation/optimized_unitcell.poscar'))
    supercell = read(rc2_dir.joinpath('first_order_optimisation/optimized_supercell.poscar'))

rc3_dir = root_dir.joinpath('rc3')
conv_rc3_extract_ifcs(unitcell = unitcell,
                          supercell = supercell,
                          sampling = sampling,
                          timestep = 1,
                          dir = rc3_dir,
                          first_order = False,
                          first_order_rc2 = None,
                          displ_threshold_firstorder = 0.000001,
                          max_iterations_first_order = 20,
                          rc2 = 14, 
                          rc3s = [1,2,3,4,5,6,7,8,9], 
                          polar = True,
                          loto_filepath = Path('loto_splitting/infile.lotosplitting'), 
                          stride = 1, 
                          temperature = temperature,
                          bin_prefix = 'mpirun',
                          tdep_bin_directory = '',
                          ifc_diff_threshold = 0.000001, # eV/A^2
                          n_rc3_to_average = 4,
                          conv_criterion_diff = 'avg')


