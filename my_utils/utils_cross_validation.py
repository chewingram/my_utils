import numpy as np
import os
import sys
from .utils import data_reader, space, path, inv_dict, mae, rmse, R2, cap_first, low_first
import .utils_mlip as mlp
import random as rnd
from ase.io import read, write
import shutil
from matplotlib import pyplot as plt
import matplotlib
from subprocess import run
import subprocess

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




def cross_validate_kfold(nfolds, mlip_bin, dataset_path, mtp_pot_params, mtp_train_params):
    '''
    Function to launch the cross validation using the k-fold validation. Currently it works with MTP.
    Args:
    nfolds(int): number of folds
    mlip_bin(str, path): path to the mlip binary
    dataset_path(str, path): path to the dataset
    mtp_pot_params(dict): dictionary with the parameters for the mtp potential file. Example:
                          dict(sp_count = 2,
                          maxd = max_dist,
                          rad_bas_sz = 8,
                          rad_bas_type = 'RBChebyshev',
                          lev = 22,
                          mtps_dir = '/scratch/users/s/l/slongo/codes/mlip-2/untrained_mtps',
                          out_name = 'init.mtp')
                          ...see mlp.make_mtp_file() function for more details on the keys. !! the keys 'mind' and 'wdir' are set
                             in this function, so no need to include them in the dictionary (they will be overwritten anyways).
    mtp_train_params(dict): dictionary with the parameters for the training. Example:
                            dict(ene_weight = 1,
                            for_weight = 5,
                            str_weight = 5,
                            bgfs_tol = 1e-3,
                            max_iter = 1000,
                            weighting = 'vibrations',
                            init_par = 'random',
                            up_mindist = False,
                            tr_pot_n = 'pot.mtp')
                        ...see mlp.train_pot() function for more details on the keys.
    
    '''
    # mlpi-2 bin
    mlip_bin = 'mpirun /scratch/ulg/matnan/slongo/codes/mlip-2/build1/mlp'

    # seed
    seed = rnd.randint(0, 99999)
    rnd.seed(seed)

    # retrieve the data
    data = read(dataset_path, index=':')

    rnd.shuffle(data)
    indices = np.array(kfold_ind(size=len(data), k=nfolds)[1])
    res_sum = f'Results of k-fold cross-validation. Seed used to shuffle the dataset: {seed}\n' 
    res_sum += f'#n. fold  \tfold size\trmse eV/at (E){space(6)}\tmae eV/at (E){space(7)}\tR2 (E){space(14)}\t'
    res_sum += f'rmse eV/Angst (F)   \tmae eV/Angst (F)    \tR2 (F){space(14)}\t'
    res_sum += f'rmse GPa (S){space(8)}\tmae GPa (S){space(9)}\tR2 (S){space(14)}\n'
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
    for i in range(len(indices) + 1):
        if i == 0: # for the first fold the lower index is 0
            i1 = 0
        else:
            i1 = indices[i-1] + 1 # for all the other folds the lower index is the upper index of the previous fold + 1
        if i == len(indices): # for the last fold the upper index is the length of the dataset - 1
            i2 = len(data) - 1
        else:
            i2 = indices[i] # for all the other folds the upper index is the i-th value of the array indeces
        mask = np.repeat(False, len(data)) # make a mask
        mask[i1:i2] = True # true only for the fold (test set)
        train_set = [x for x, bool in zip(data, mask) if not bool]
        test_set = [x for x, bool in zip(data, mask) if bool]
        set_lengths.append(len(test_set))
        #print(f'N. confs. of the train set: {len(train_set)}; test set: {i1}-{i2} (n. confs: {len(test_set)})')

        # Now we have train- and test set. We need to train and then test.

        dir = path("tmp/")
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)


        mlp.conv_ase_to_mlip2(train_set, f'{dir}TrainSet.cfg')
        #write_cfg(f'{dir}TrainSet.cfg', train_set, ['Mo', 'S'])
        mlp.conv_ase_to_mlip2(test_set, f'{dir}TestSet.cfg')
        #write_cfg(f'{dir}TestSet.cfg', test_set, ['Mo', 'S'])


         # compute the minimum distance
        min_dist = mlp.find_min_dist(train_set)
        mtp_pot_params['mind'] = min_dist
        
        mtp_pot_params['wdir'] = dir

        mlp.make_mtp_file(**mtp_pot_params)
        #shutil.copyfile('init.mtp', f'{dir}init.mtp')

        # Training phase
        tmp_params = dict(mlip_bin = mlip_bin,
                         init_path = f'{dir}init.mtp',
                         train_set_path = f'{dir}TrainSet.cfg',
                         dir = dir,
                         params = mtp_train_params)

        mlp.train_pot(**tmp_params)

        #cmd = f'{mlip_bin} train init.mtp TrainSet.cfg --trained-pot-name=pot.mtp'
        #run(cmd.split(), cwd=dir)
        #os.system(f'cp pot.mtp {dir}')

        # Compute e-f-s on the training set
        efs_params = dict(mlip_bin = mlip_bin, 
                          confs_path = f'{dir}TrainSet.cfg', 
                          pot_path = f'{dir}pot.mtp', 
                          out_path = f'{dir}ResTrain.cfg',
                          dir = dir)
        print(f'Giving to calc_efs: out_path = {efs_params["out_path"]}')
        mlp.calc_efs(**efs_params)

        #cmd = f'{mlip_bin} calc-efs pot.mtp TrainSet.cfg ResTrain.cfg'
        #run(cmd.split(), cwd=dir)


        # Compute errors and make comparison files
        errs_train = mlp.make_comparison(f'{dir}TrainSet.cfg', f'{dir}ResTrain.cfg', make_file=False, dir='./', outfile_pref='MLIP-')

        # Testing phase (computing e-f-s on the test set)
        efs_params['confs_path'] = f'{dir}TestSet.cfg'
        efs_params['out_path'] = f'{dir}ResTest.cfg'
        mlp.calc_efs(**efs_params)
        #cmd = f'{mlip_bin} calc-efs pot.mtp TestSet.cfg ResTest.cfg'
        #run(cmd.split(), cwd=dir)

        # Compute errors and male comparison files
        errs_test = mlp.make_comparison(f'{dir}TestSet.cfg', f'{dir}ResTest.cfg', make_file=False, dir='./', outfile_pref='Test-')


        #for x in [0,1,2]:
        #    plot_correlations(dtset_ind=0, dir=dir, ind=x, offsets=[0.025, 2, 20], save=False)

        # Save errors to do the summary of the errors
        e_rmse.append(errs_test['energy'][0])
        e_mae.append(errs_test['energy'][1])
        e_R2.append(errs_test['energy'][2])
        f_rmse.append(errs_test['forces'][0])
        f_mae.append(errs_test['forces'][1])
        f_R2.append(errs_test['forces'][2])
        s_rmse.append(errs_test['stress'][0])
        s_mae.append(errs_test['stress'][1])
        s_R2.append(errs_test['stress'][2])

    # Complete and save the summary of errors
    res_sum += f'max values\t{space(9)}\t{max(e_rmse):<20.10f}\t{max(e_mae):<20.10f}\t{max(e_R2):<20.10f}\t{max(f_rmse):<20.10f}\t{max(f_mae):<20.10f}\t{max(f_R2):<20.10f}\t{max(s_rmse):<20.10f}\t{max(s_mae):<20.10f}\t{max(s_R2):<20.10f}\n'
    for i in range(len(e_rmse)):
        res_sum += f"{i+1:<10}\t{set_lengths[i]:<9}\t{e_rmse[i]:<20.10f}\t{e_mae[i]:<20.10f}\t{e_R2[i]:<20.10f}\t{f_rmse[i]:<20.10f}\t{f_mae[i]:<20.10f}\t{f_R2[i]:<20.10f}\t{s_rmse[i]:<20.10f}\t{s_mae[i]:<20.10f}\t{s_R2[i]:<20.10f}\n"
    res_sum_name = 'res_summary.dat'
    with open(res_sum_name, 'w') as fl:
        fl.write(res_sum)
