import numpy as np
import os
from pathlib import Path
from time import time

from ase.io import read, write

from my_utils.utils_mlip import find_min_dist, train_pot_from_ase_tmp
from my_utils.mlip_models.mtp_model import MTP_model



dataset = read('Dataset.traj', index=':')

min_dist = find_min_dist(dataset)       

mlip_bin = '/Users/samuel/Work/codes/mlip-2/build1/mlp'
untrained_pot_file_dir = '/Users/samuel/Work/codes/mlip-2/untrained_mtps'

MTP = MTP_model(mtp_level=6,
                min_dist = min_dist,
                max_dist = 8,
                radial_basis_size = 8,
                radial_basis_type = 'RBChebyshev',
                train_set=dataset)


train_params = dict(ene_weight = 1,
                    for_weight = 1,
                    str_weight = 1,
                    bfgs_tol = 1e-3,
                    max_iter = 500,
                    weighting = 'structures',
                    init_par = 'random',
                    up_mindist = False,
                    trained_pot_name = 'pot.mtp')

MTP.train(mlip_bin=mlip_bin,
            untrained_pot_file_dir = untrained_pot_file_dir,
            training_dir=Path('./'),
            bin_pref='mpirun -n 6',
            final_evaluation=True,
            params=train_params)


