import numpy as np
import os
from pathlib import Path

from ase.io import read, write
from my_utils.utils_cross_validation import cross_validate_kfold, check_convergence_kfold
from my_utils.utils_cross_validation_parallel import launch_parallel_k_fold, parallel_conv_crossvalidation

nfolds = 4
mlip_bin = '/Users/samuel/Work/codes/mlip-2/build1/mlp'
mpirun = 'mpirun -n 6'
dataset = read('Trajectory.traj', index=':')
train_flag_params = dict(ene_weight = 1,
                         for_weight = 1,
                         str_weight = 1,
                         sc_b_for = 0,
                         max_iter = 200,
                         bfgs_tol = 1e-3,
                         weighting = 'structures',
                         init_par = 'random',
                         up_mindist=False)

train_params = dict(untrained_pot_file_dir = '/Users/samuel/Work/codes/mlip-2/untrained_mtps/',
                    mtp_level = 6,
                    max_dist = 6,
                    radial_basis_size = 6,
                    radial_basis_type = 'RBChebyshev')


parallel_conv_crossvalidation(root_dir='./',
                              job_file_path='./job.sh',
                              increase_step=20,
                              nfolds=4,
                              min_dtsize=1,
                              mpirun=mpirun,
                              mlip_bin=mlip_bin, 
                              dataset=dataset, 
                              train_flag_params=train_flag_params,
                              train_params=train_params)


# launch_parallel_k_fold(nfolds,
#                            mpirun,
#                            mlip_bin,
#                            dataset,
#                            train_flag_params,
#                            train_params,
#                            job_template_path='./job.sh',
#                            logging=True,
#                            logger_name='paral_k_launcher',
#                            logger_filepath='paral_k.log',
#                            debug_log=False)
