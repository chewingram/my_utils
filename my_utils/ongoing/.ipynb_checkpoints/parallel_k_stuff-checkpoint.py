import os
from pathlib import Path
import pickle as pkl
from copy import deepcopy as cp
import datetime
import shutil
import random as rnd
import numpy as np

from ase.io import read, write

from ..utils_cross_validation import kfold_ind

def parallel_conv_crossvalidation(root_dir='./',
                                  jobfile_path='./job.sh',
                                  increase_step=10,
                                  nfolds=10,
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

    
    root_dir = Path('./')
    
    # make directory for the iterations
    iters_dir = root_dir.joinpath('iterations')
    if iters_dir.isdir():
        shutil.rmtree(iters_dir.absolute())
    iters_dir.mkdirs(parents=True, exist_ok=True)
    
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

    print(f'{n_iters} crossvalidations ({nfolds}-fold) will be prepared')
    print(f'The dataset has {dtsize} elements. Small datasets will be used by increasing their size by {increase_step}.')

    
    
    msg = f'Launching the crossvalidation with structures from n. 0 to n. {i}'
    msg += f' (iteration n. {i+1})'
    
    print(msg)

    if offset == 0:
        print(f'The minimum size is {min_dtsize}')
    else:
        msg = f'The minimum size is {min_dtsize}, but we need to include the first {offset} configuration to make the size'
        msg += f' of the dataset a multiple of the increasing step.'
        print(msg)

    for i in range(n_iters):
        # make dir for the current iteration
        iter_dir = iters_dir.joinpath(f'i_iter')
        if iter_dir.isdir():
            shutil.rmtree(iter_dir.absolute())
        iter_dir.mkdirs(parents=True, exist_ok=True)
        
        curr_dataset = dataset[:offset+min_dtsize-1 + i*increase_step +1]
        
        launch_parallel_k_fold(wdir=iter_dir.absolute(),
                               nfolds=nfolds,
                               mpirun=mpirun,
                               mlip_bin=mlip_bin,
                               dataset=curr_dataset, 
                               train_flag_params=train_flag_params,
                               train_params=train_params,
                               job_template_path=job_template_path,
                               logging=True, 
                               logger_name='paral_k_logger', 
                               logger_filepath='paral_k.log', 
                               debug_log=False)

    

    
    
    

def single_k(wdir):
    
    wdir = Path(wdir)
    
    l1 = setup_logging(logger_name='single_k_log', log_file=wdir.joinpath('single_k_log'), debug=False)
    
    # first we need to import everything
    with open('parameters.pkl', 'rb') as fl:
        import_parameters = pkl.load(fl)

    # content:
    # explanatory variable
    # mpirun
    # mlip_bin
    # train_flag_params
    # train_params
    # dataset_path
    # i1
    # i2
    
    mpirun = import_parameters['mpirun']
    mlip_bin = import_parameters['mlip_bin']
    train_flag_params = import_parameters['train_flag_params']
    train_params = import_parameters['train_params']
    traj_path = Path(import_parameters['dataset_path'])
    i1 = import_parameters['i1']
    i2 = import_parameters['i2']
    
    # starting message
    msg = f'Single k-fold run started on ' + datetime.datetime.now().strftime("%d %b %Y - %H:%M:%S") + '.'
    msg += f'\nThe slice of dataset {i1}:{i2} will be used as test set, the rest as training set.'
    
    dataset = read(traj_path, index=':')
    mask = np.repeat(False, len(dataset)) # make a mask
    mask[i1:i2] = True # true only for the fold (test set)
    train_set = [cp(x) for x, bool in zip(dataset, mask) if not bool]
    test_set = [cp(x) for x, bool in zip(dataset, mask) if bool]
    set_length = len(test_set)

    # Now we have train- and test set. We need to train and then test.
     # compute the minimum distance
    min_dist = mlp.find_min_dist(train_set)
    
    trained_pot_name = 'pot.mtp'

    train_flag_params['trained_pot_name'] = trained_pot_name
    if 'cur_pot_n' in train_flag_params.keys(): del train_flag_params['cur_pot_n']
    train_params['min_dist'] = min_dist
    train_params['dir'] = wdir.absolute()
    train_params['params'] = train_flag_params
    train_params['train_set'] = train_set
    train_params['mlip_bin'] = mlip_bin
    train_params['mpirun'] = mpirun
    train_params['final_evaluation'] = False

    if 'val_cfg' in train_params.keys(): del train_params['val_cfg']
    
    l1.info(f'  -About to begin the training phase')
    mlp.train_pot_from_ase_tmp(**train_params)
    l1.info(f'  -Training done')
                                          
    ml_train_set = mlp.calc_efs_from_ase(mlip_bin = mlip_bin, 
                                         atoms=train_set, 
                                         mpirun=train_params['mpirun'], 
                                         pot_path=wdir.joinpath('pot.mtp').absolute(), 
                                         cfg_files=False, 
                                         dir=wdir.absolute(),
                                         write_conf=False, 
                                         outconf_name=None)
    
    errs_train = mlp.make_comparison(is_ase1=True,
                                     is_ase2=True,
                                     structures1=train_set, 
                                     structures2=ml_train_set, 
                                     props='all', 
                                     make_file=False, 
                                     dir=wdir.absolute(),
                                     outfile_pref='', 
                                     units=None)
    
    
    l1.info(f'  -About to begin the testing phase')
    ml_test_set = mlp.calc_efs_from_ase(mlip_bin = train_params['mlip_bin'], 
                                        atoms=test_set, 
                                        mpirun=train_params['mpirun'], 
                                        pot_path=wdir.joinpath('pot.mtp').absolute(), 
                                        cfg_files=False, 
                                        out_path='./out.cfg',
                                        dir=wdir.absolute(),
                                        write_conf=False, 
                                        outconf_name=None)
                                    
    errs_test = mlp.make_comparison(is_ase1=True,
                                     is_ase2=True,
                                     structures1=test_set, 
                                     structures2=ml_test_set, 
                                     props='all', 
                                     make_file=False, 
                                     dir=wdir.absolute(),
                                     outfile_pref='', 
                                     units=None)
    
    res_header = f'#n. fold  fold size  {space(3)}rmse eV/at (E)  {space(4)}mae eV/at (E)  {space(11)}R2 (E)  '
    res_header += f'{space(0)}rmse eV/Angst (F)  {space(1)}mae eV/Angst (F)  {space(11)}R2 (F)  '
    res_header += f'{space(5)}rmse eV/Angst^2 (S)  {space(6)}mae eV/Angst^2 (S)  {space(11)}R2 (S)\n'
    
    l1.info(f'  -Testing done.\n  Results for the fold:')
    l1.info('  ' + res_header)

    e_rmse = errs_test['energy'][0]               
    e_mae = errs_test['energy'][1]
    e_R2 = errs_test['energy'][2]                 
    f_rmse = errs_test['forces'][0]
    f_mae = errs_test['forces'][1]                   
    f_R2 = errs_test['forces'][2]                   
    s_rmse = errs_test['stress'][0]                    
    s_mae = errs_test['stress'][1]            
    s_R2 = errs_test['stress'][2]
    
    res_text = f"{1:>8}  {set_length:>9}  {e_rmse:>17.10f}  {e_mae:>17.10f}  {e_R2:>17.10f}  " + \
                  f"{f_rmse:>17.10f}  {f_mae:>17.10f}  {f_R2:>17.10f}  {s_rmse:>17.10f}  {s_mae:>17.10f}  " + \
                  f"{s_R2:>17.10f}"
     
    l1.info('  ' + res_text)

    with open(wdir.joinpath('resfile.pkl'), 'wb') as fl:
        pkl.dump([errs_train, errs_test], fl)


def launch_parallel_k_fold(wdir,
                           nfolds,
                           mpirun,
                           mlip_bin,
                           dataset, 
                           train_flag_params,
                           train_params,
                           job_template_path='./job.sh',
                           logging=True, 
                           logger_name='paral_k_logger', 
                           logger_filepath='paral_k.log', 
                           debug_log=False):
    
    wdir = Path('./')
    folds_dir = wdir.joinpath('folds')
    if folds_dir.is_dir():
        shutil.rmtree(folds_dir.absolute())
    folds_dir.mkdir(parents=True, exist_ok=True)
    job_template_path = Path(job_template_path)
    
    run_single_k_filepath = Path(__file__).parent.joinpath('../data/parallel_k_fold/run_single_k.py').resolve().absolute()
    
    # seed
    seed = rnd.randint(0, 99999)
    rnd.seed(seed)
    
    print(f'Shuffling the dataset with seed = {seed}') 
    dataset = cp(dataset)
    rnd.shuffle(dataset)
    indices = np.array(kfold_ind(size=len(dataset), k=nfolds)[1])

    set_lengths = []
    
    print(f'{len(indices) + 1} folds will be defined')
    
    for i in range(len(indices) + 1):
        if i == 0: # for the first fold the lower index is 0
            i1 = 0
        else:
            i1 = indices[i-1] + 1 # for all the other folds the lower index is the upper index of the previous fold + 1
        if i == len(indices): # for the last fold the upper index is the length of the dataset - 1
            i2 = len(dataset) - 1
        else:
            i2 = indices[i] # for all the other folds the upper index is the i-th value of the array indeces
        
        fold_dir = folds_dir.joinpath(f'{i+1}_fold')
        fold_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = wdir.joinpath('Trajectory.traj')
        write(dataset_path.absolute(),dataset)

        explanatory_var = ''
        import_parameters = dict(mpirun=mpirun,
                                 mlip_bin=mlip_bin,
                                 train_flag_params=train_flag_params,
                                 train_params=train_params,
                                 dataset_path=dataset_path.absolute(),
                                 i1=i1,
                                 i2=i2,
                                 seed=seed,
                                )
        parameters_file_path = fold_dir.joinpath('parameters.pkl') 
        fetcher_path = Path(__file__).parent.joinpath('../data/parallel_k_fold/fetch_results.py').resolve().absolute()
        with open(parameters_file_path, 'wb') as fl:
            pkl.dump(import_parameters, fl)
        shutil.copy(job_template_path, fold_dir)
        shutil.copy(run_single_k_filepath, fold_dir)
        shutil.copy(fetcher_path, wdir)


