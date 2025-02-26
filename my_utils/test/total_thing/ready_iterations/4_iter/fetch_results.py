from pathlib import Path
import numpy as np
import pickle as pkl
from my_utils.utils import space


root_dir = Path('./')
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

with open(root_dir.joinpath('cross_validation_results.pkl'), 'wb') as fl:
    pkl.dump(var_to_save, fl)
