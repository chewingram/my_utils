import os
import sys
from subprocess import run
import subprocess
from copy import deepcopy as cp
from pathlib import Path
from math import ceil
import logging
import datetime
import inspect
import shutil
import pickle as pkl
import random as rnd

from matplotlib import pyplot as plt
import matplotlib

import numpy as np

from .utils import setup_logging, mute_logger, data_reader, space, path, inv_dict, mae, rmse, R2, cap_first, low_first
from . import utils_mlip as mlp

from ase.io import read, write




def kfold_ind(size, k):
    '''
    Given the size of a dataset (size) and the number of folds (k), return a list containing the indices of the dataset
    of the last configuration of the corresponding fold. E.g. [11, 23] for a dataset of 34 confs means that
    there are three folds: the first is 0th-11th, the second fold is 12th-23rd and the last one is 24th-33rd. If size is not
    a mutiple of k, after distributing int(size/k) confs to each fold, an extra conf is given to the first
    size%k folds.
    Arguments:
    size(int): size of the dataset
    k(int): number of folds
    '''
    fold_sizes = []
    for i in range(k):
        fold_sizes.append(int(size/k))
        if size%k != 0 and i < size%k:
            fold_sizes[-1] += 1
    ind_list = [fold_sizes[0] - 1]
    for i in range(1, len(fold_sizes) - 1):
        ind_list.append(ind_list[-1] + fold_sizes[i])
    return fold_sizes, ind_list    



def plot_convergence_stuff(res, dt_sizes, dir):
    dir = Path(dir)
    if not dir.is_dir():
        dir.mkdir(parents=True)
    
    # DATA PROCESSING
    train_errs_ene = np.array([x[0]['energy'] for x in res]) # shape (n_iterations, n_metrics, n_folds)
    train_errs_for = np.array([x[0]['forces'] for x in res])
    train_errs_str = np.array([x[0]['stress'] for x in res])

    test_errs_ene = np.array([x[1]['energy'] for x in res])
    test_errs_for = np.array([x[1]['forces'] for x in res])
    test_errs_str = np.array([x[1]['stress'] for x in res])



    # Error analysis metrics for training
    train_metrics = {
        'RMSE': 0,  # Index for RMSE in the third dimension
        'MAE': 1,   # Index for MAE in the third dimension
        'R2': 2     # Index for R2 in the third dimension
    }

    # Initialize dictionaries for average and maximum metrics
    avg_train_metrics = {}
    max_train_metrics = {}

    avg_test_metrics = {}
    max_test_metrics = {}

    # Loop through the metrics and compute averages and maxima
    for metric_name, metric_idx in train_metrics.items():
        avg_train_metrics[metric_name] = np.array([
            train_errs_ene[:, metric_idx, :].mean(axis=1),
            train_errs_for[:, metric_idx, :].mean(axis=1),
            train_errs_str[:, metric_idx, :].mean(axis=1),
        ])
        if metric_name == 'R2':
            max_train_metrics[metric_name] = np.array([
                train_errs_ene[:, metric_idx, :].min(axis=1),
                train_errs_for[:, metric_idx, :].min(axis=1),
                train_errs_str[:, metric_idx, :].min(axis=1),
            ])

            max_test_metrics[metric_name] = np.array([
                test_errs_ene[:, metric_idx, :].min(axis=1),
                test_errs_for[:, metric_idx, :].min(axis=1),
                test_errs_str[:, metric_idx, :].min(axis=1),
            ])
        else:
            max_train_metrics[metric_name] = np.array([
                train_errs_ene[:, metric_idx, :].max(axis=1),
                train_errs_for[:, metric_idx, :].max(axis=1),
                train_errs_str[:, metric_idx, :].max(axis=1),
            ])

            max_test_metrics[metric_name] = np.array([
                test_errs_ene[:, metric_idx, :].max(axis=1),
                test_errs_for[:, metric_idx, :].max(axis=1),
                test_errs_str[:, metric_idx, :].max(axis=1),
            ])

        avg_test_metrics[metric_name] = np.array([
            test_errs_ene[:, metric_idx, :].mean(axis=1),
            test_errs_for[:, metric_idx, :].mean(axis=1),
            test_errs_str[:, metric_idx, :].mean(axis=1),
        ])
        
    ### ACTUAL PLOTTING
    
    #### RMSE
    # plot avg RMSE
    y_e = avg_test_metrics['RMSE'][0]
    y_f = avg_test_metrics['RMSE'][1]
    y_s = avg_test_metrics['RMSE'][2]

    fig_avg_rmse, ax1 = plt.subplots(3, 1, figsize=(10, 10) )
    fig_avg_rmse.suptitle('Average RMSE (over the k-fold crossvalidation iterations)')

    ax1[0].set_xlabel('Size of the dataset')
    ax1[0].set_ylabel('Energy RMSE (eV/at)')
    ax1[0].plot(dt_sizes, y_e, label='Energy RMSE', ls='-', marker='o', color='blue')

    ax1[1].set_xlabel('Size of the dataset')
    ax1[1].set_ylabel('Forces RMSE (ev/Angs)')
    ax1[1].plot(dt_sizes, y_f, label='Forces RMSE', ls='-', marker='o', color='red')


    ax1[2].set_xlabel('Size of the dataset')
    ax1[2].set_ylabel('Stress RMSE (GPa)')
    ax1[2].plot(dt_sizes, y_s, label='Stress RMSE', ls='-', marker='o', color='green')

    fig_avg_rmse.patch.set_linewidth(1)
    fig_avg_rmse.patch.set_edgecolor('black')


    # plot max RMSE
    y_e = max_test_metrics['RMSE'][0]
    y_f = max_test_metrics['RMSE'][1]
    y_s = max_test_metrics['RMSE'][2]


    fig_max_rmse, ax2 = plt.subplots(3, 1, figsize=(10, 10) )
    fig_max_rmse.suptitle('Maximum RMSE (over the k-fold crossvalidation iterations)')

    ax2[0].set_xlabel('Size of the dataset')
    ax2[0].set_ylabel('Energy RMSE (eV/at)')
    ax2[0].plot(dt_sizes, y_e, label='Energy RMSE', ls='-', marker='o', color='blue')

    ax2[1].set_xlabel('Size of the dataset')
    ax2[1].set_ylabel('Forces RMSE (ev/Angs)')
    ax2[1].plot(dt_sizes, y_f, label='Forces RMSE', ls='-', marker='o', color='red')


    ax2[2].set_xlabel('Size of the dataset')
    ax2[2].set_ylabel('Stress RMSE (GPa)')
    ax2[2].plot(dt_sizes, y_s, label='Stress RMSE', ls='-', marker='o', color='green')

    fig_max_rmse.patch.set_linewidth(1)
    fig_max_rmse.patch.set_edgecolor('black')
    
    
    #### MAE
    # plot avg MAE
    y_e = avg_test_metrics['MAE'][0]
    y_f = avg_test_metrics['MAE'][1]
    y_s = avg_test_metrics['MAE'][2]

    fig_avg_mae, ax1 = plt.subplots(3, 1, figsize=(10, 10) )
    fig_avg_mae.suptitle('Average MAE (over the k-fold crossvalidation iterations)')

    ax1[0].set_xlabel('Size of the dataset')
    ax1[0].set_ylabel('Energy MAE (eV/at)')
    ax1[0].plot(dt_sizes, y_e, label='Energy MAE', ls='-', marker='o', color='blue')

    ax1[1].set_xlabel('Size of the dataset')
    ax1[1].set_ylabel('Forces MAE (ev/Angs)')
    ax1[1].plot(dt_sizes, y_f, label='Forces MAE', ls='-', marker='o', color='red')


    ax1[2].set_xlabel('Size of the dataset')
    ax1[2].set_ylabel('Stress MAE (GPa)')
    ax1[2].plot(dt_sizes, y_s, label='Stress MAE', ls='-', marker='o', color='green')

    fig_avg_mae.patch.set_linewidth(1)
    fig_avg_mae.patch.set_edgecolor('black')


    # plot max MAE
    y_e = max_test_metrics['MAE'][0]
    y_f = max_test_metrics['MAE'][1]
    y_s = max_test_metrics['MAE'][2]


    fig_max_mae, ax2 = plt.subplots(3, 1, figsize=(10, 10) )
    fig_max_mae.suptitle('Maximum MAE (over the k-fold crossvalidation iterations)')

    ax2[0].set_xlabel('Size of the dataset')
    ax2[0].set_ylabel('Energy MAE (eV/at)')
    ax2[0].plot(dt_sizes, y_e, label='Energy MAE', ls='-', marker='o', color='blue')

    ax2[1].set_xlabel('Size of the dataset')
    ax2[1].set_ylabel('Forces MAE (ev/Angs)')
    ax2[1].plot(dt_sizes, y_f, label='Forces MAE', ls='-', marker='o', color='red')


    ax2[2].set_xlabel('Size of the dataset')
    ax2[2].set_ylabel('Stress MAE (GPa)')
    ax2[2].plot(dt_sizes, y_s, label='Stress MAE', ls='-', marker='o', color='green')

    fig_max_mae.patch.set_linewidth(1)
    fig_max_mae.patch.set_edgecolor('black')
    
    
    #### R2
    # plot avg R2
    y_e = avg_test_metrics['R2'][0]
    y_f = avg_test_metrics['R2'][1]
    y_s = avg_test_metrics['R2'][2]

    fig_avg_R2, ax1 = plt.subplots(3, 1, figsize=(10, 10) )
    fig_avg_R2.suptitle('Average R$^2$ (over the k-fold crossvalidation iterations)')

    ax1[0].set_xlabel('Size of the dataset')
    ax1[0].set_ylabel('Energy R$^2$')
    ax1[0].plot(dt_sizes, y_e, label='Energy R2', ls='-', marker='o', color='blue')

    ax1[1].set_xlabel('Size of the dataset')
    ax1[1].set_ylabel('Forces R$^2$')
    ax1[1].plot(dt_sizes, y_f, label='Forces R$^2$', ls='-', marker='o', color='red')


    ax1[2].set_xlabel('Size of the dataset')
    ax1[2].set_ylabel('Stress R$^2$')
    ax1[2].plot(dt_sizes, y_s, label='Stress R$^2$', ls='-', marker='o', color='green')

    fig_avg_R2.patch.set_linewidth(1)
    fig_avg_R2.patch.set_edgecolor('black')


    # plot min R2
    y_e = max_test_metrics['R2'][0]
    y_f = max_test_metrics['R2'][1]
    y_s = max_test_metrics['R2'][2]


    fig_min_R2, ax2 = plt.subplots(3, 1, figsize=(10, 10) )
    fig_min_R2.suptitle('Minimum R$^2$ (over the k-fold crossvalidation iterations)')

    ax2[0].set_xlabel('Size of the dataset')
    ax2[0].set_ylabel('Energy R$^2$')
    ax2[0].plot(dt_sizes, y_e, label='Energy R$^2$', ls='-', marker='o', color='blue')

    ax2[1].set_xlabel('Size of the dataset')
    ax2[1].set_ylabel('Forces R$^2$')
    ax2[1].plot(dt_sizes, y_f, label='Forces R$^2$', ls='-', marker='o', color='red')


    ax2[2].set_xlabel('Size of the dataset')
    ax2[2].set_ylabel('Stress R$^2$')
    ax2[2].plot(dt_sizes, y_s, label='Stress R$^2$', ls='-', marker='o', color='green')

    fig_min_R2.patch.set_linewidth(1)
    fig_min_R2.patch.set_edgecolor('black')


    fig_avg_rmse.savefig(bbox_inches='tight', dpi=600, format='png', fname=dir.joinpath('avg_rmse.png').resolve())
    fig_max_rmse.savefig(bbox_inches='tight', dpi=600, format='png', fname=dir.joinpath('max_rmse.png').resolve())
    fig_avg_mae.savefig(bbox_inches='tight', dpi=600, format='png', fname=dir.joinpath('avg_mae.png').resolve())
    fig_max_mae.savefig(bbox_inches='tight', dpi=600, format='png', fname=dir.joinpath('max_mae.png').resolve())
    fig_avg_R2.savefig(bbox_inches='tight', dpi=600, format='png', fname=dir.joinpath('avg_R2.png').resolve())
    fig_min_R2.savefig(bbox_inches='tight', dpi=600, format='png', fname=dir.joinpath('min_R2.png').resolve())
    

def cross_validate_kfold(nfolds,
                         mpirun,
                         mlip_bin,
                         dataset, 
                         train_flag_params,
                         train_params,
                         logging=True, 
                         logger_name='cross_val_k_logger', 
                         logger_filepath='kfold_crossvalidation.log', 
                         debug_log=False):
    '''
    Function to launch the cross validation using the k-fold validation. Currently it works with MTP.
    Args:
    nfolds: int
        number of folds
    mpirun: str
        command for mpi or similar (e.g. 'mpirun')
    mlip_bin: str, path
        path to the mlip binary
    dataset: ase.Atoms
        dataset as an ASE trajectory
    train_flag_params: dict
        dictionary containing the flags to use; these are the possibilities:
        ene_weight: float, default=1
            weight of energies in the fitting
        for_weight: float, default=0.01
            weight of forces in the fitting
        str_weight: float, default=0.001 
            weight of stresses in the fitting
        sc_b_for: float, default=0
            if >0 then configurations near equilibrium (with roughtly force < 
            <double>) get more weight
        val_cfg: str 
            filename with configuration to validate
        max_iter: int, default=1000
            maximal number of iterations
        bfgs_tol: float, default=1e-3
            stop if error dropped by a factor smaller than this over 50 BFGS 
            iterations
        weighting: {'vibrations', 'molecules', 'structures'}, default=vibrations 
            how to weight configuration wtih different sizes relative to each 
            other
        init_par: {'random', 'same'}, default='random'
            how to initialize parameters if a potential was not pre-fitted;
            - random: random initialization
            - same: this is when interaction of all species is the same (more 
                    accurate fit, but longer optimization)
        skip_preinit: bool 
            skip the 75 iterations done when parameters are not given
        up_mindist: bool
            updating the mindist parameter with actual minimal interatomic 
            distance in the training set
    train_params: dict
        dictionary with the parameters for the training; these are the parameters to set:
        untrained_pot_file_dir: str 
            path to the directory containing the untrained mtp init files (.mtp)
        mtp_level: {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 24, 26, 28}
            level of the mtp model to train
        max_dist: float
                cutoff radius for the radial part (unit: Angstrom)
        radial_basis_size: int, default=8
            number of basis functions to use for the radial part
        radial_basis_type: {'RBChebyshev', ???}, default='RBChebyshev'
            type of basis functions to use for the radial part
    logging: bool; default = True
        activate the logging
    logger_name: str; default = None
        name of the logger to use; if a logger with that name already exists (e.g. it was 
        created by a calling function) it will be used, otherwise a nre one will be created.
        If the logger_name is None, the root logger will be used.
    logger_filepath: str; default = 'convergence.log'
        path to the file that will contain the log. If None and no preexisting file handler is there, 
        no log file will be created. If an existing logger name is provided and it already has
        a file handler, this argument will be ignored and the preexisting file handler will be used; if no
        handler already exists, or a new logger is created, this argument is the filepath of the log file.
     debug_log: bool;  default = False
         if a new logger is create (preexisting_logger_name = None), then activate debug logging
     
     Returns
     -------
     errs_train: list of dict
         list containing the errors on the training set for each fold-iteration; for a given fold-iteration,
         the dictionary keys are the names of the properties {'energy', 'stress', 'forces'}, while the values
         are lists [rmse, mae, R2] for each property.
     errs_test: dict
         same thing as errs_train, but for the test set.
    
    '''
    
    # set the logger 
    if logging == True:
        # set logger
        l1 = setup_logging(logger_name=logger_name, log_file=logger_filepath, debug=debug_log)
    else:
        l1 = mute_logger()
    
    start_message = 'K-fold crossvalidation begun at ' + datetime.datetime.now().strftime("%d %b %Y - %H:%M:%S")
    l1.info(start_message)
    
    # seed
    seed = rnd.randint(0, 99999)
    rnd.seed(seed)
    
    l1.info(f'Shuffling the dataset with seed = {seed}') 
    dataset = cp(dataset)
    rnd.shuffle(dataset)
    indices = np.array(kfold_ind(size=len(dataset), k=nfolds)[1])
    res_sum = f'Results of k-fold cross-validation. Seed used to shuffle the dataset: {seed}\n' 
    res_header = f'#n. fold  fold size  {space(3)}rmse eV/at (E)  {space(4)}mae eV/at (E)  {space(11)}R2 (E)  '
    res_header += f'{space(0)}rmse eV/Angst (F)  {space(1)}mae eV/Angst (F)  {space(11)}R2 (F)  '
    res_header += f'{space(5)}rmse GPa (S)  {space(6)}mae GPa (S)  {space(11)}R2 (S)\n'
    res_sum += res_header
    
    tr_e_rmse = []
    tr_e_mae = []
    tr_e_R2 = []
    tr_f_rmse = []
    tr_f_mae = []
    tr_f_R2 = []
    tr_s_rmse = []
    tr_s_mae = []
    tr_s_R2 = []
    
    e_rmse = []
    e_mae = []
    e_R2 = []
    f_rmse = []
    f_mae = []
    f_R2 = []
    s_rmse = []
    s_mae = []
    s_R2 = []
    
    set_lengths = []
    
    l1.info(f'{len(indices) + 1} folds will be defined')
    
    for i in range(len(indices) + 1):
        l1.info(f'# Fold n. {i+1} of {len(indices) + 1} just started')
        if i == 0: # for the first fold the lower index is 0
            i1 = 0
        else:
            i1 = indices[i-1] + 1 # for all the other folds the lower index is the upper index of the previous fold + 1
        if i == len(indices): # for the last fold the upper index is the length of the dataset - 1
            i2 = len(dataset) - 1
        else:
            i2 = indices[i] # for all the other folds the upper index is the i-th value of the array indeces
        mask = np.repeat(False, len(dataset)) # make a mask
        mask[i1:i2] = True # true only for the fold (test set)
        train_set = [cp(x) for x, bool in zip(dataset, mask) if not bool]
        test_set = [cp(x) for x, bool in zip(dataset, mask) if bool]
        set_lengths.append(len(test_set))
        #print(f'N. confs. of the train set: {len(train_set)}; test set: {i1}-{i2} (n. confs: {len(test_set)})')

        # Now we have train- and test set. We need to train and then test.

        dir = Path(f"tmp/iter_{i}")
        if os.path.exists(dir.absolute()):
            shutil.rmtree(dir.absolute())
        dir.mkdir(parents=True, exist_ok=True)


#         mlp.conv_ase_to_mlip2(train_set, f'{dir}TrainSet.cfg')
#         #write_cfg(f'{dir}TrainSet.cfg', train_set, ['Mo', 'S'])
#         mlp.conv_ase_to_mlip2(test_set, f'{dir}TestSet.cfg')
#         #write_cfg(f'{dir}TestSet.cfg', test_set, ['Mo', 'S'])


         # compute the minimum distance
        min_dist = mlp.find_min_dist(train_set)
        
        trained_pot_name = 'pot.mtp'

        train_flag_params['trained_pot_name'] = trained_pot_name
        if 'cur_pot_n' in train_flag_params.keys(): del train_flag_params['cur_pot_n']
        train_params['min_dist'] = min_dist
        train_params['dir'] = dir.absolute()
        train_params['params'] = train_flag_params
        train_params['train_set'] = train_set
        train_params['mlip_bin'] = mlip_bin
        train_params['mpirun'] = mpirun

        if 'val_cfg' in train_params.keys(): del train_params['val_cfg']
        
        l1.info(f'  -About to begin the training phase')
        mlp.train_pot_from_ase_tmp(**train_params)
        l1.info(f'  -Training done')
                                              
        ml_train_set = mlp.calc_efs_from_ase(mlip_bin = mlip_bin, 
                                             atoms=train_set, 
                                             mpirun=train_params['mpirun'], 
                                             pot_path=dir.joinpath('pot.mtp').absolute(), 
                                             cfg_files=False, 
                                             dir=dir.absolute(),
                                             write_conf=False, 
                                             outconf_name=None)
        
        errs_train = mlp.make_comparison(is_ase1=True,
                                         is_ase2=True,
                                         structures1=train_set, 
                                         structures2=ml_train_set, 
                                         props='all', 
                                         make_file=False, 
                                         dir=dir.absolute(),
                                         outfile_pref='', 
                                         units=None)
        
        
        l1.info(f'  -About to begin the testing phase')
        ml_test_set = mlp.calc_efs_from_ase(mlip_bin = train_params['mlip_bin'], 
                                            atoms=test_set, 
                                            mpirun=train_params['mpirun'], 
                                            pot_path=dir.joinpath('pot.mtp').absolute(), 
                                            cfg_files=False, 
                                            out_path='./out.cfg',
                                            dir=dir.absolute(),
                                            write_conf=False, 
                                            outconf_name=None)
                                        
        errs_test = mlp.make_comparison(is_ase1=True,
                                         is_ase2=True,
                                         structures1=test_set, 
                                         structures2=ml_test_set, 
                                         props='all', 
                                         make_file=False, 
                                         dir=dir.absolute(),
                                         outfile_pref='', 
                                         units=None)
        l1.info(f'  -Testing done.\n  Results for the fold n. {i+1}:')
        l1.info('  ' + res_header)
        
        # Save errors to do the summary of the errors
        
        tr_e_rmse.append(errs_train['energy'][0])                
        tr_e_mae.append(errs_train['energy'][1])                
        tr_e_R2.append(errs_train['energy'][2])                   
        tr_f_rmse.append(errs_train['forces'][0])                    
        tr_f_mae.append(errs_train['forces'][1])                   
        tr_f_R2.append(errs_train['forces'][2])                   
        tr_s_rmse.append(errs_train['stress'][0])                    
        tr_s_mae.append(errs_train['stress'][1])            
        tr_s_R2.append(errs_train['stress'][2])
        
        e_rmse.append(errs_test['energy'][0])                
        e_mae.append(errs_test['energy'][1])                
        e_R2.append(errs_test['energy'][2])                   
        f_rmse.append(errs_test['forces'][0])                    
        f_mae.append(errs_test['forces'][1])                   
        f_R2.append(errs_test['forces'][2])                   
        s_rmse.append(errs_test['stress'][0])                    
        s_mae.append(errs_test['stress'][1])            
        s_R2.append(errs_test['stress'][2])
        
        res_text = f"{i+1:>8}  {set_lengths[i]:>9}  {e_rmse[i]:>17.10f}  {e_mae[i]:>17.10f}  {e_R2[i]:>17.10f}  " + \
                  f"{f_rmse[i]:>17.10f}  {f_mae[i]:>17.10f}  {f_R2[i]:>17.10f}  {s_rmse[i]:>17.10f}  {s_mae[i]:>17.10f}  " + \
                  f"{s_R2[i]:>17.10f}"
        
        res_sum += res_text     
        l1.info('  ' + res_text)
        
        l1.info(f'  Current fold done')
        
        res_sum += f'\n'
        
        
    tr_e_rmse = np.array(tr_e_rmse, dtype='float')
    tr_e_mae = np.array(tr_e_mae, dtype='float')
    tr_e_R2 = np.array(tr_e_R2, dtype='float')
    tr_f_rmse = np.array(tr_f_rmse, dtype='float')
    tr_f_mae = np.array(tr_f_mae, dtype='float')
    tr_f_R2 = np.array(tr_f_R2, dtype='float')
    tr_s_rmse = np.array(tr_s_rmse, dtype='float')
    tr_s_mae = np.array(tr_s_mae, dtype='float')
    tr_s_R2 = np.array(tr_s_R2, dtype='float')
    
    e_rmse = np.array(e_rmse, dtype='float')
    e_mae = np.array(e_mae, dtype='float')
    e_R2 = np.array(e_R2, dtype='float')
    f_rmse = np.array(f_rmse, dtype='float')
    f_mae = np.array(f_mae, dtype='float')
    f_R2 = np.array(f_R2, dtype='float')
    s_rmse = np.array(s_rmse, dtype='float')
    s_mae = np.array(s_mae, dtype='float')
    s_R2 = np.array(s_R2, dtype='float')
    
    res_text = f'         max values  {space(5)}{max(e_rmse):<17.10f}  {max(e_mae):<17.10f}  {max(e_R2):<17.10f}  '+ \
              f'{max(f_rmse):<17.10f}  {max(f_mae):<17.10f}  {max(f_R2):<17.10f}  {max(s_rmse):<17.10f}  '+ \
              f'{max(s_mae):<17.10f}  {max(s_R2):<17.10f}\n'
    res_sum += res_text


    res_text = f'         min values  {space(5)}{min(e_rmse):<17.10f}  {min(e_mae):<17.10f}  {min(e_R2):<17.10f}  '+ \
              f'{min(f_rmse):<17.10f}  {min(f_mae):<17.10f}  {min(f_R2):<17.10f}  {min(s_rmse):<17.10f}  '+ \
              f'{min(s_mae):<17.10f}  {min(s_R2):<17.10f}\n'
    res_sum += res_text


    res_text = f'     average values  {space(5)}{e_rmse.mean():<17.10f}  {e_mae.mean():<17.10f}  '+ \
              f'{e_R2.mean():<17.10f}  {f_rmse.mean():<17.10f}  {f_mae.mean():<17.10f}  '+ \
              f'{f_R2.mean():<17.10f}  {s_rmse.mean():<17.10f}  {s_mae.mean():<17.10f}  {s_R2.mean():<17.10f}\n'
    res_sum += res_text
    
    
    l1.info(res_sum)

    # Complete and save the summary of errors
#     for i in range(len(e_rmse)):
#         res_sum += f"{i+1:<10}  {set_lengths[i]:<9}  {e_rmse[i]:<20.10f}  {e_mae[i]:<20.10f}  {e_R2[i]:<20.10f}  {f_rmse[i]:<20.10f}  {f_mae[i]:<20.10f}  {f_R2[i]:<20.10f}  {s_rmse[i]:<20.10f}  {s_mae[i]:<20.10f}  {s_R2[i]:<20.10f}\n"
    
#     res_sum += f'max values  {space(9)}  {max(e_rmse):<20.10f}  {max(e_mae):<20.10f}  {max(e_R2):<20.10f}  {max(f_rmse):<20.10f}  {max(f_mae):<20.10f}  {max(f_R2):<20.10f}  {max(s_rmse):<20.10f}  {max(s_mae):<20.10f}  {max(s_R2):<20.10f}\n'
    
#     res_sum += f'min values  {space(9)}  {min(e_rmse):<20.10f}  {min(e_mae):<20.10f}  {min(e_R2):<20.10f}  {min(f_rmse):<20.10f}  {min(f_mae):<20.10f}  {min(f_R2):<20.10f}  {min(s_rmse):<20.10f}  {min(s_mae):<20.10f}  {min(s_R2):<20.10f}\n'
    
#     res_sum += f'average values  {space(9)}  {e_rmse.mean():<20.10f}  {e_mae.mean():<20.10f}  {e_R2.mean():<20.10f}  {f_rmse.mean():<20.10f}  {f_mae.mean():<20.10f}  {f_R2.mean():<20.10f}  {s_rmse.mean():<20.10f}  {s_mae.mean():<20.10f}  {s_R2.mean():<20.10f}\n'
    
    res_sum_name = Path('res_summary.dat')
    with open(res_sum_name.absolute(), 'w') as fl:
        fl.write(res_sum)
    
    train_res = {}
    train_res['energy'] = np.array([tr_e_rmse, tr_e_mae, tr_e_R2])
    train_res['forces'] = np.array([tr_f_rmse, tr_f_mae, tr_f_R2])
    train_res['stress'] = np.array([tr_s_rmse, tr_s_mae, tr_s_R2])
    
    test_res = {}
    test_res['energy'] = np.array([e_rmse, e_mae, e_R2])
    test_res['forces'] = np.array([f_rmse, f_mae, f_R2])
    test_res['stress'] = np.array([s_rmse, s_mae, s_R2])
    
    var_to_return = [train_res, test_res]
    
    # shape of var_to_return: n_set_types, n_properties, n_metrics, n_folds
    # These are the dimensions: 1-[training set, test set]; 2-dict['energy', 'forces', 'stress']; 3-[rmse, mae, r2]; 4-fold
    return var_to_return

        
def check_convergence_kfold(increase_step,
                            nfolds,
                            min_dtsize=1,
                            mpirun='',
                            mlip_bin='mlp', 
                            dataset=None, 
                            train_flag_params=None,
                            train_params=None, 
                            logging=True, 
                            logger_name=None,
                            logger_filepath='convergence.log', 
                            debug_log=False,
                            check_for_restart=False):
    '''This function checks if a dataset is converged with a specified MTP model.
    
    It is assumed that the order in the dataset is the same of the hypotetical convergence, in other words, 
    the convergence of the potential is checked with respect to the size of the dataset as it increases along
    the trajectory.
    Increasing subsets of the dataset are used to train and test the potential according to a k-fold protocol, 
    where k (nfolds) is kept constant, while the size of the folds increases through the convergence check.
    
    Parameters
    ----------
    increase_step: int
        number of structures to increase the dataset by at each iteration of the training; it must be a multiple of nfolds
    nfolds: int
        number of folds used in the k-fold crossvalidation protocol  
    min_dtsize: int
        minimum size of the initial (smallest) dataset used
    mpirun: str
        command for mpi or similar (e.g. 'mpirun')
    mlip_bin: str, path
        path to the mlip binary
    dataset: ase.Atoms
        dataset as an ASE trajectory
    train_flag_params: dict
        dictionary containing the flags to use; these are the possibilities:
        ene_weight: float, default=1
            weight of energies in the fitting
        for_weight: float, default=0.01
            weight of forces in the fitting
        str_weight: float, default=0.001 
            weight of stresses in the fitting
        sc_b_for: float, default=0
            if >0 then configurations near equilibrium (with roughtly force < 
            <double>) get more weight
        val_cfg: str 
            filename with configuration to validate
        max_iter: int, default=1000
            maximal number of iterations
        bfgs_tol: float, default=1e-3
            stop if error dropped by a factor smaller than this over 50 BFGS 
            iterations
        weighting: {'vibrations', 'molecules', 'structures'}, default=vibrations 
            how to weight configuration wtih different sizes relative to each 
            other
        init_par: {'random', 'same'}, default='random'
            how to initialize parameters if a potential was not pre-fitted;
            - random: random initialization
            - same: this is when interaction of all species is the same (more 
                    accurate fit, but longer optimization)
        skip_preinit: bool 
            skip the 75 iterations done when parameters are not given
        up_mindist: bool
            updating the mindist parameter with actual minimal interatomic 
            distance in the training set
    train_params: dict
        dictionary with the parameters for the training; these are the parameters to set:
        untrained_pot_file_dir: str 
            path to the directory containing the untrained mtp init files (.mtp)
        mtp_level: {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 24, 26, 28}
            level of the mtp model to train
        max_dist: float
                cutoff radius for the radial part (unit: Angstrom)
        radial_basis_size: int, default=8
            number of basis functions to use for the radial part
        radial_basis_type: {'RBChebyshev', ???}, default='RBChebyshev'
            type of basis functions to use for the radial part
    logging: bool; default = True
        activate the logging
    logger_name: str; default = None
        name of the logger to use; if a logger with that name already exists (e.g. it was 
        created by a calling function) it will be used, otherwise a new one will be created.
        If the logger_name is None, the root logger will be used.
    logger_filepath: str; default = 'convergence.log'
        path to the file that will contain the log. If None and no preexisting file handler is there, 
        no log file will be created. If an existing logger name is provided and it already has
        a file handler, this argument will be ignored and the preexisting file handler will be used; if no
        handler already exists, or a new logger is created, this argument is the filepath of the log file.
     debug_log: bool;  default = False
         if a new logger is created (preexisting_logger_name = None), then activate debug logging
    
    '''

    ### AUXILIARY FUNCTIONS ###
    def find_new_path(path):
        parent = path.parent
        name = path.name[:-4] # exclude the extension and dot
        j = 0
        i = 0
        while i == 0:
            new_path = parent.joinpath(f'{name}_{j}.pkl')
            if new_path.is_file():
                j += 1
            else:
                break
        return new_path
    ###########################
    
    if logging == True:
        # set logger
        l1 = setup_logging(logger_name=logger_name, log_file=logger_filepath, debug=debug_log)
    else:
        l1 = mute_logger()

    dt_sizes = [] 
    res = []
    n_iterations_done = 0
    
    tmp = [x for x in inspect.signature(check_convergence_kfold).parameters]
    parameters = {x:y for x,y in [(z,w) for z,w in locals().items() if z in tmp]}
    
    results_filepath = Path('./kfold_convergence_results.pkl')
    
    if check_for_restart == True:
        if not results_filepath.is_file():
            msg = f'No restarting point (e.g. file named kfold_convergence_results.pkl) found in {results_filepath.parent}\n'
            l1.info(msg)
            is_restart = False  
        else:
            with open(results_filepath.absolute(), 'rb') as fl:
                restart_file = pkl.load(fl)
            dt_sizes = restart_file[1] # overwrite the empty lisy
            res = restart_file[2] # overwrite the empty list
            old_parameters = restart_file[3][0]
            n_iterations_done = len(dt_sizes) # overwrite 0
            old_dtsize = len(old_parameters['dataset'])
            old_min_dtsize = old_parameters['min_dtsize']
            old_increase_step = old_parameters['increase_step']
            old_n_iters = int((old_dtsize-old_min_dtsize)/old_increase_step) + 1
            if n_iterations_done < old_n_iters:
                msg = f'A restarting point was found ({results_filepath.absolute()}) with {n_iterations_done} iteration(s) done '
                msg += f'out of {old_n_iters}.\nThe process will be restarted from the next iteration, reusing the parameters passed to the function '
                msg +=f'in the previous call (the new ones will be ignored).'
                l1.info(msg)
                is_restart = True
                parameters = old_parameters
            else:
                newpath = find_new_path(results_filepath)
                results_filepath.rename(newpath)
                msg = f'A restarting point was found ({results_filepath.absolute()}) but it is already completed ({n_iterations_done} iterations)'
                msg += f'\nThe file was renamed {new_path.name} and the process will be restarted from scratch.'
                l1.info(msg)
                is_restart = False
                n_iterations_done = 0 # it must be given back the original value of 0
    else:
        is_restart = False

    if is_restart == True:
        # if it is a restart, update the variables with the new parameters (which are actually the old ones)
        increase_step = parameters['increase_step']
        nfolds = parameters['nfolds']
        min_dtsize = parameters['min_dtsize']
        mpirun = parameters['mpirun']
        mlip_bin = parameters['mlip_bin']
        dataset = parameters['dataset']
        train_flag_params = parameters['train_flag_params']
        train_params = parameters['train_params']
        # we use the logging settings passed to the function, irrespective of any restart
        #logging = parameters['logging']
        #logger_name = parameters['logger_name']
        #logger_filepath = parameters['logger_filepath'] 
        #debug_log = parameters['debug_log']


    l1.info('Evaluation of the convergence of dataset started on ' + datetime.datetime.now().strftime("%d %b %Y - %H:%M:%S"))
    dtsize = len(dataset)
    
    
    if min_dtsize < 1:
        raise ValueError('min_dtsize must be greater than 0!')
    if increase_step%nfolds != 0:
        raise ValueError('The increasing step must be a multiple of the number of folds!')
    if dataset == None:
        raise ValueError('You must provide a dataset!')
    if train_flag_params == None:
        raise ValueError('You must provide the training flag paramters!')
    if train_params == None:
        raise ValueError('You must provide the training parameters!')
        
    n_iters = int((dtsize-min_dtsize)/increase_step) + 1 # number of iterations to run to evaluate convergence
    offset = (dtsize-min_dtsize)%increase_step

    l1.info(f'{n_iters-n_iterations_done} crossvalidations ({nfolds}-fold) will be launched')
    l1.info(f'The dataset has {dtsize} elements. Small datasets will be used by increasing their size by {increase_step}.')

    if is_restart == False:
        if offset == 0:
            l1.info(f'The minimum size is {min_dtsize}')
        else:
            msg = f'The minimum size is {min_dtsize}, but we need to include the first {offset} configuration to make the size'
            msg += f' of the dataset a multiple of the increasing step.'
            l1.info(msg)
    else:
        if offset == 0:
            l1.info(f'!Just for the record! - The minimum size is {min_dtsize}')
        else: 
            msg = f'!Just for the record! - The minimum size is {min_dtsize}, but we needed to include the first {offset} '
            msg += f'configuration to make the size'
            msg += f' of the dataset a multiple of the increasing step.'
            l1.info(msg)
    
    expl = "This file contains four elements:"
    expl += "\n- this explanatory variable;"
    expl += "\n- dt_sizes: a list with the size of the total datasets used in each crossvalidation;"
    expl += "\n- res: contains for each iteration two elements: errs_train and errs_test as they are output by mlp.make_comparison"
    expl += "\n\t its shape is: "
    expl += "\n\t\t(n_iteration, n_set_types, n_properties, n_metrics, n_folds):
    expl += "\n\t and these are the dimensions:"
    expl += "\n\t\t1-iteration; 2-[training set, test set]; 3-dict['energy', 'forces', 'stress']; 4-[rmse, mae, r2]; 5-fold;"
    expl += "\n- meta: a list of two elements: (i) the parameters that were passed to the function when it was launched last time, "
    expl += "and (ii) the number of iterations done before finishing or being interrupted."
    
    for i in range(n_iterations_done, n_iters):
        msg = f'Launching the crossvalidation with structures from n. 0 to n. {offset+min_dtsize-1 + i*increase_step}'
        msg += f' (iteration n. {i+1})'
        l1.info(msg)
        
        curr_dataset = dataset[:offset+min_dtsize-1 + i*increase_step +1]
        
        res.append(cross_validate_kfold(nfolds, 
                                        mpirun,
                                        mlip_bin, 
                                        curr_dataset,
                                        train_flag_params, 
                                        train_params, 
                                        logging=True,
                                        logger_name=l1.name))
        dt_sizes.append(len(curr_dataset))
        l1.info('Crossvalidation done\n')

        meta = [parameters, i+1]
        with open(results_filepath, 'wb') as fl:
            pkl.dump([expl, dt_sizes, res, meta], fl)
            
        
    # res shape: (n_iteration, n_set_types, n_properties, n_metrics, n_folds)
    # These are the dimensions:
    # 1-iteration; 2-[training set, test set]; 3-dict['energy', 'forces', 'stress']; 4-[rmse, mae, r2]; 5-fold
        
    plot_convergence_stuff(res, dt_sizes, dir='Convergence_figures/')
