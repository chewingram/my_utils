import os
from pathlib import Path
import pickle as pkl
from copy import deepcopy as cp
import datetime
import shutil
import random as rnd
import numpy as np
import subprocess

from ase.io import read, write

from .utils_cross_validation import kfold_ind
from .utils import setup_logging, mute_logger, data_reader, space, path, inv_dict, mae, rmse, R2, cap_first, low_first
from . import utils_mlip as mlp



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
    # conf_index
    # seed
    
    mpirun = import_parameters['mpirun']
    mlip_bin = import_parameters['mlip_bin']
    train_flag_params = import_parameters['train_flag_params']
    train_params = import_parameters['train_params']
    traj_path = Path(import_parameters['dataset_path'])
    i1 = import_parameters['i1']
    i2 = import_parameters['i2']
    conf_index = import_parameters['conf_index']
    seed = import_parameters['seed']

    rnd.seed(seed)

    # starting message
    msg = f'Single k-fold run started on ' + datetime.datetime.now().strftime("%d %b %Y - %H:%M:%S") + '.'
    msg += f'\nThe slice of dataset {i1}:{i2} will be used as test set, the rest as training set.'
    
    dataset = read(traj_path, index=':')[:conf_index + 1]
    rnd.shuffle(dataset)

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
                           dataset_path, 
                           conf_index,
                           tot_dtsize,
                           train_flag_params,
                           train_params,
                           job_file_path='./job.sh',
                           logging=True, 
                           logger_name='paral_k_logger', 
                           logger_filepath='paral_k.log', 
                           debug_log=False):
    
    dataset_path = Path(dataset_path)
    dtsize = conf_index + 1

    folds_dir = wdir.joinpath('folds')
    if folds_dir.is_dir():
        shutil.rmtree(folds_dir.absolute())
    folds_dir.mkdir(parents=True, exist_ok=True)
    job_file_path = Path(job_file_path)
    
    run_single_k_filepath = Path(__file__).parent.joinpath('data/parallel_k_fold/run_single_k.py').resolve().absolute()
    
    # seed
    seed = rnd.randint(0, 99999)
    rnd.seed(seed)
    
    #print(f'Shuffling the dataset with seed = {seed}') 
    #dataset = cp(dataset)
    #rnd.shuffle(dataset)
    indices = np.array(kfold_ind(size=dtsize, k=nfolds)[1])

    set_lengths = []
    
    print(f'{len(indices) + 1} folds will be defined')
    
    for i in range(len(indices) + 1):
        if i == 0: # for the first fold the lower index is 0
            i1 = 0
        else:
            i1 = indices[i-1] + 1 # for all the other folds the lower index is the upper index of the previous fold + 1
        if i == len(indices): # for the last fold the upper index is the length of the dataset - 1
            i2 = dtsize - 1
        else:
            i2 = indices[i] # for all the other folds the upper index is the i-th value of the array indeces
        
        fold_dir = folds_dir.joinpath(f'{i+1}_fold')
        fold_dir.mkdir(parents=True, exist_ok=True)

        #dataset_path = wdir.joinpath('Trajectory.traj')
        #write(dataset_path.absolute(),dataset)
        explanatory_var = ''
        import_parameters = dict(mpirun=mpirun,
                                 mlip_bin=mlip_bin,
                                 train_flag_params=train_flag_params,
                                 train_params=train_params,
                                 dataset_path=dataset_path.absolute(),
                                 conf_index=conf_index,
                                 i1=i1,
                                 i2=i2,
                                 seed=seed,
                                )
        parameters_file_path = fold_dir.joinpath('parameters.pkl') 
        with open(parameters_file_path, 'wb') as fl:
            pkl.dump(import_parameters, fl)
        shutil.copy(job_file_path, fold_dir)
        shutil.copy(run_single_k_filepath, fold_dir)


def parallel_conv_crossvalidation(root_dir='./',
                                  job_file_path='./job.sh',
                                  increase_step=10,
                                  nfolds=10,
                                  min_dtsize=1,
                                  mpirun='',
                                  mlip_bin='mlp', 
                                  dataset=None, 
                                  train_flag_params=None,
                                  train_params=None,
                                  sbatch=False):

    
    root_dir = Path('./')
    job_file_path = Path(job_file_path) 
    # make directory for the iterations
    iters_dir = root_dir.joinpath('iterations')
    if iters_dir.is_dir():
        shutil.rmtree(iters_dir.absolute())
    iters_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = iters_dir.joinpath('Dataset.traj')
    write(dataset_path, dataset)
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

    
    if offset == 0:
        print(f'The minimum size is {min_dtsize}')
    else:
        msg = f'The minimum size is {min_dtsize}, but we need to include the first {offset+1} configuration to make the size'
        msg += f' of the dataset a multiple of the increasing step.'
        print(msg)

    for i in range(n_iters):            
        msg = f'Setting the crossvalidation with structures from n. 0 to n. {offset+min_dtsize-1 + i*increase_step}'
        msg += f' (iteration n. {i+1})'
        print(msg)

        # make dir for the current iteration
        iter_dir = iters_dir.joinpath(f'{i+1}_iter')
        if iter_dir.is_dir():
            shutil.rmtree(iter_dir.absolute())
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        #curr_dataset = dataset[:offset+min_dtsize-1 + i*increase_step +1]
        conf_index = offset+min_dtsize-1 + i*increase_step # index of the last (included) configuration of the current dataset
        launch_parallel_k_fold(wdir=iter_dir.absolute(),
                               nfolds=nfolds,
                               mpirun=mpirun,
                               mlip_bin=mlip_bin,
                               dataset_path=dataset_path.absolute(), 
                               conf_index=conf_index,
                               tot_dtsize=dtsize,
                               train_flag_params=train_flag_params,
                               train_params=train_params,
                               job_file_path=job_file_path.absolute(),
                               logging=True, 
                               logger_name='paral_k_logger', 
                               logger_filepath='paral_k.log', 
                               debug_log=False)
    fetcher_path = Path(__file__).parent.joinpath('data/parallel_k_fold/fetch_results.py').resolve().absolute()
    shutil.copy(fetcher_path, root_dir)
    
    if sbatch == True:
        for i in range(n_iters):            
            # make dir for the current iteration
            iter_dir = iters_dir.joinpath(f'{i+1}_iter')
            list_of_dirs = sorted(iter_dir.glob('folds/*_fold'))
            for dir in list_of_dirs:
                job_file_path = dir.joinpath('job.sh')
                subprocess.run(f'sbatch {job_file_path.absolute()}', shell=True, cwd=dir.absolute())
                



def fetch_results_single_crossv(root_dir):
    root_dir = Path(root_dir)
    list_of_folders = sorted(root_dir.joinpath('folds').glob('*_fold'))

    res_header = f'#n. fold  fold size  {space(5)}rmse eV/at (E)  {space(6)}mae eV/at (E)  {space(13)}R2 (E)  '
    res_header += f'{space(2)}rmse eV/Angst (F)  {space(3)}mae eV/Angst (F)  {space(13)}R2 (F)  '
    res_header += f'{space(0)}rmse eV/Angst^2 (S)  {space(1)}mae eV/Angst^2 (S)  {space(13)}R2 (S)\n'
    res_sum = res_header

    seeds = []
    set_lengths = []
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

    for i, folder in enumerate(list_of_folders):
        with open(folder.joinpath('resfile.pkl'), 'rb') as fl:
            data = pkl.load(fl)
        errs_train = data[0]
        errs_test = data[1]

        # we also need to retrieve the size of the fold used as test set and the seed
        with open(folder.joinpath('parameters.pkl'), 'rb') as fl:
            data = pkl.load(fl)
            i1 = data['i1']
            i2 = data['i2']
            set_lengths.append(i2-i1)
            seeds.append(data['seed'])

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

        res_text = f"{i+1:>8}  {set_lengths[i]:>9}  {e_rmse[i]:>19.10f}  {e_mae[i]:>19.10f}  {e_R2[i]:>19.10f}  " + \
                f"{f_rmse[i]:>19.10f}  {f_mae[i]:>19.10f}  {f_R2[i]:>19.10f}  {s_rmse[i]:>19.10f}  {s_mae[i]:>19.10f}  " + \
                f"{s_R2[i]:>19.10f}"

        res_sum += res_text
        res_sum += '\n'

    if not all([x==seeds[0] for x in seeds]):
        raise ValueError('Two or more folds have different seed!')
    seed = seeds[0]

    top_txt = f'Results of k-fold cross-validation. Seed used to shuffle the dataset: {seed}\n'
    res_sum = top_txt + res_sum

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


    res_sum_name = root_dir.joinpath('res_summary.dat')
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

    var_to_save = [train_res, test_res]

    #with open(root_dir.joinpath('cross_validation_results.pkl'), 'wb') as fl:
    #    pkl.dump(var_to_save, fl)
    return var_to_save