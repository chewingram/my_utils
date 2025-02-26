from pathlib import Path
import numpy as np
import pickle as pkl
from my_utils.utils_cross_validation import plot_convergence_stuff
from my_utils.utils_cross_validation_parallel import fetch_results_single_crossv


iters_dir = Path('./iterations')
list_of_folders = sorted(iters_dir.glob('*_iter'))

res = []
dt_sizes = []

for i, folder in enumerate(list_of_folders):
    
    res.append(fetch_results_single_crossv(root_dir=folder))

    # we need to retrieve the size of the dataset used in this iteration
    with open(folder.joinpath('folds/1_fold/parameters.pkl'), 'rb') as fl:
        data = pkl.load(fl)
    dt_size = data['conf_index']
    dt_sizes.append(dt_size)

expl_var = ''

with open('convergence_kfold_crossv.pkl', 'wb') as fl:
    pkl.dump([expl_var, res, dt_sizes], fl)

# res shape: (n_iteration, n_set_types, n_properties, n_metrics, n_folds)
# These are the dimensions:
# 1-iteration; 2-[training set, test set]; 3-dict['energy', 'forces', 'stress']; 4-[rmse, mae, r2]; 5-fold
    
plot_convergence_stuff(res, dt_sizes, dir='./Convergence_figures/')

