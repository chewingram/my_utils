from my_utils.utils_tdep import convergence_tdep_stride_or_sampling_size
from pathlib import Path
import numpy as np

mult_mat = np.array([[4, 4, 0],
                     [-2, 2, 0],
                     [0, 0, 3]])

for temperature in range(100, 800, 100):
    root_dir = Path(f'./T{temperature}K')
    print(root_dir.joinpath(f'T{temperature}K.traj').absolute())
    convergence_tdep_stride_or_sampling_size(stride=True,
                                             sampling_size=False,
                                             temperature = temperature,
                                             root_dir=root_dir,
                                             tdep_bin_directory = Path('/gpfs/scratch/ehpc14/ulie583683/venvs/env_3.12.1/codes/tdep/bin/'),
                                             bin_prefix = 'mpirun',
                                             first_order = True,
                                             displ_threshold_firstorder = 0.000001,
                                             max_iterations_first_order = 20,
                                             nthrow = 0,
                                             rc2 = 20,
                                             rc3 = 10,
                                             ts = 1,
                                             max_stride=4,
                                             stride_step=1,
                                             uc_path = root_dir.joinpath(f'T{temperature}K_unitcell.json').absolute(),
                                             mult_mat = mult_mat,
                                             traj_path = root_dir.joinpath(f'T{temperature}K.traj').absolute(),
                                             polar = True,
                                             loto_filepath = root_dir.joinpath('loto_splitting/infile.lotosplitting'),
                                             job=True,
                                             job_template='./job.sh')
    
    


