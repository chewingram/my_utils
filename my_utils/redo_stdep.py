import numpy as np
import os
import shutil
from pathlib import Path

from my_utils import utils_mlip as mlp
from my_utils import utils_tdep as tdp
from my_utils.utils import min_distance_to_surface, ln_s_f
from ase.io import read, write
from ase.build import make_supercell as mk_supcercell


# sTDEP steps:
# 1. first iteration
#     1.1 sample 4 configurations with the initial ifcs 
#     1.2 compute properties
#     1.3 extract ifcs: first order thing + convergence w.r.t rc2
# 2. loop over the iterations
#     2.i.1 sample N(i) configurations with the previous ifcs
#     2.i.2 compute properties
#     2.i.3 extract ifcs - convergence w.r.t rc2
#     2.i.4 assess convergence
# 3. final tidying of the files and sampling


root_dir = './' #
ucell = read('unitcell.poscar') #
make_supercell = True #
scell_mat = np.eye(3)*3 #
T = 300 #
ifc_guess = False #
max_freq = 15 #
quantum = True #
tdep_bin_directory = None #
loto = False #
preexisting_ifcs = False #
pref_bin = '' #
mlip_pot_path = '/Users/samuel/Work/ML/MoS2/mlip/mtp_3/training/pot.mtp' #
mlip_bin = '/Users/samuel/Work/codes/mlip-2/build1/mlp' #
first_order = False #
displ_threshold_firstorder = 0.0001 #
max_iterations_first_order = 20 #
rc3 = None #
loto_filepath = None #
polar = False #
niters = 5 #
max_err_threshold = 0.00001 #




if make_supercell == False:
    if scell is None:
        raise TypeError('Since make_supercell is False, you must provide a supercell')
else:
    if scell_mat is None:
        raise TypeError('Since make_supercell is True, you must provide scell_mat!')
    else:
        scell = mk_supcercell(ucell, scell_mat)
if loto == True:
    if loto_path is None:
        raise TypeValue('Since loto is True, you must provide loto_path!')
    loto_path = Path(loto_path)
    if not loto_path.is_file():
        raise ValueError(f'The file {loto_path.absolute()} does not exist!')
    loto_cmd = 'testo_comando'
if preexisting_ifcs is False:
    if max_freq is None:
        raise TypeError('Since preexisting_ifcs is False, you must provide max_freq!')
else:
    preexisting_ifcs_path = Path(preexisting_ifcs_path)
    if not preexisting_ifcs_path.is_file():
        raise ValueError(f'The file {preexisting_ifcs_path.absolute()} does not exist!')
    max_freq = False
if tdep_bin_directory is not None:
    tdep_bin_directory = Path(tdep_bin_directory)
    




root_dir = Path(root_dir)

# infiles
infiles_dir = root_dir.joinpath('infiles')
infiles_dir.mkdir(parents=True, exist_ok=True)
write(infiles_dir.joinpath('infile.ucposcar'), ucell, format='vasp')
write(infiles_dir.joinpath('infile.ssposcar'), scell, format='vasp')
if loto == True:
    shutil.copy(loto_path, infiles_dir.joinpath('infile.lotosplitting'))

if preexisting_ifcs == True:
    shutil.copy(preexisting_ifcs_path, infiles_dir.joinpath('infile.forceconstant'))

iters_dir = root_dir.joinpath('iterations')
iters_dir.mkdir(parents=True, exist_ok=True)
nconfs = [4]
nconfs.extend([x*20 for x in range(1,7)])

print('++++++++++++++++++++++++++++++++++++++')
print('----------- sTDEP launched -----------')
print('++++++++++++++++++++++++++++++++++++++')
# 1. first iteration
for iter in range(1, niters+1):
    print(f'====== ITERATION n. {iter} ======')
    iter_dir = iters_dir.joinpath(f'iter_{iter}')
    iter_dir.mkdir(parents=True, exist_ok=True)

    #   1.1 sample 4 configurations with the initial ifcs
    make_canonical_configurations_parameters = dict(ucell = ucell,
                                                    scell = scell,
                                                    nconf = nconfs[iter-1],
                                                    temp = T,
                                                    quantum = quantum,
                                                    dir = iter_dir,
                                                    outfile_name = 'new_confs.traj', # this will be saved inside dir
                                                    pref_bin=pref_bin,
                                                    tdep_bin_directory=tdep_bin_directory)
    
    if iter == 1:
        if preexisting_ifcs == False:
            make_canonical_configurations_parameters['max_freq'] = max_freq
        else:
            make_canonical_configurations_parameters['ifcfile_path'] = preexisting_ifcs_path
    else:
        make_canonical_configurations_parameters['ifcfile_path'] = last_ifc_path 
    
    tdp.make_canonical_configurations(**make_canonical_configurations_parameters)

    latest_confs = read(iter_dir.joinpath('new_confs.traj'), index=':')

    #   1.2 1.2 compute properties
    prop_iter = iter_dir.joinpath('true_props')
    prop_iter.mkdir(exist_ok=True, parents=True)

    latest_confs_computed = mlp.calc_efs_from_ase(mlip_bin = mlip_bin, 
                                                  atoms = latest_confs, 
                                                  mpirun = 'mpirun -n 6', 
                                                  pot_path = mlip_pot_path, 
                                                  cfg_files=False, 
                                                  dir = prop_iter,
                                                  write_conf = True, 
                                                  outconf_name = 'new_confs_computed.traj')
    latest_confs_computed_path = prop_iter.joinpath('new_confs_computed.traj')

    ifc_dir = iter_dir.joinpath('ifc')
    ifc_dir.mkdir(parents=True, exist_ok=True)
    
    min_dist = min([min_distance_to_surface(x.get_cell()) for x in latest_confs_computed]) ##
    print('min dist')
    print(min_dist)
    rc2s = [x for x in range(int(max(min_dist - 10, 0)), int(min_dist+1))] #
    print(f'rc2s: {rc2s}')
    
    last_ifc_path, max_diffs, avg_diffs = tdp.conv_rc2_extract_ifcs(unitcell = ucell,
                                                                      supercell = scell,
                                                                      sampling = latest_confs_computed,
                                                                      timestep = 1,
                                                                      dir = ifc_dir,
                                                                      first_order = first_order,
                                                                      displ_threshold_firstorder = displ_threshold_firstorder,
                                                                      max_iterations_first_order = max_iterations_first_order,
                                                                      rc2s = rc2s, 
                                                                      rc3 = rc3, 
                                                                      polar = polar,
                                                                      loto_filepath = loto_filepath, 
                                                                      stride = 1, 
                                                                      temperature = T,
                                                                      bin_prefix = pref_bin,
                                                                      tdep_bin_directory = tdep_bin_directory,
                                                                      max_err_threshold = max_err_threshold)
    print(f'The converged one is {last_ifc_path}')
    print(f'============================')

    

    



