import os
import numpy as np
import argparse
from copy import deepcopy as cp
from pathlib import Path
from ase.io import read, write, Trajectory
from ase.build import make_supercell
from ase.calculators.lammpsrun import LAMMPS
from mlacs.state import LammpsState
from mlacs.utilities.io_lammps import reconstruct_mlmd_trajectory
import sys
from .utils import path, min_distance_to_surface, mic_sign
from .utils_mlip import calc_efs_from_ase, pot_from_pair_style
import builtins    


original_print = builtins.print

def cprint(txt):
    original_print(txt, flush=True)
    
    


def run_md(ismpi = False,
           mpirun = 'mpirun',
           temperature = None,
           dt = 1,
           nsteps = 10_000,
           loginterval = 100,
           nthrow = 20,
           nproc = 1,
           ucell_path = None,
           mult_mat = None,
           pair_style = None,
           pair_coeff = None,
           lmp_bin = None,
           ase_lammps_command = None,
           wdir = './',
           make_wdir = True,
           NPT = True,
           NVT = True,
           nvt_scell = None,
           logfile = 'mlmd.log',
           trajfile = 'mlmd.traj',
           iso = None):

    '''
    Function to run a MD (either NPT or NVT) with LAMMPS and MLACS.
    Arguments:
    ismpi(bool): True = parallel, False = serial
    
    mpirun(str): mpirun binary
    
    temperature(float): temperature in K
    
    dt(float): timestep in fs
    
    nsteps(int): total number of timesteps done
    
    loginterval(int): how often to log lammps
    
    nthrow(int): how many timesteps to skip when saving the trajectory 
    
    nproc=(int): number of processors used; if impi = False, it is set to 1
    
    ucell_path(str): path to the unitcell file in a format AND WITH AN EXTENSION known by ASE (e.g. .poscar, .json)
    
    mult_mat(numpy array): 3D numpy array with the matrix to create a supercell; by default it's the identity
    
    pair_style(str): pair_style for lammps; e.g. "mlip /scratch/ulg/matnan/slongo/Work/ML/MoS2/MTP/MLIP/files_for_MD/mlip.ini"
    
    pair_coeff(list): pair_coeff for lammps; e.g. ['* *']
    
    mlp_bin(str): path to the mlip-2 binary; this will only be used to compute the efs of the trajectory, after its creation
                  (done with MD), while the calculator of the MD is the one installed with the LAMMPS-mlip2 interface.
    
    lmp_bin(str): path to the lammps binary; make sure it's compatible with ismpi
    
    ase_lammps_command(str): command to launch the lammps run; by default it is built as "{mpirun} -n {proc} {lmp_bin}"
    
    wdir(str): working directory
    
    make_wdir(bool): True = the wdir is created (mkdir -p); False the wdir is not created (if it doesn't exist errors may rise)
    
    NPT(bool): True = the NPT calculation is done
    
    NVT(bool): True = the NVT calculation is done; if NPT is also True, NPT is run before NVT
    
    nvt_scell(ase.Atoms or None): SUPERCELL (not unitcell + mat_mult) to use instead of mult_mat @ ucell for NVT; if None 
                                  (default) and both NPT and NVT are True, it is build with the thermalised cell coming from 
                                  the NPT. If only NVT is True, a thermalized unit cell will be needed with path 
                                  "{wdir}T{temperature}K_unitcell.json", so make sure it's there.
                                  Setting nvt_scell != None can be useful if one needs to run two independent NPT and NVT in
                                  the same working directory.
                                  
    logfile(str): name of the log file; it will be saved in the lammps workdir (NPT or NVT)
    
    trajfile(str): name of the trajectory file; it will be saved in the lammps workdir (NPT or NVT).
    
    iso(bool): mandatory when npt=True, useless otherwise. If True, then the NPT MD is done enforcing isotropy; if False, then anisotropy.
    '''
    
    if ucell_path is not None:
        ucell_path = Path(ucell_path)
    if lmp_bin is not None:
        lmp_bin = Path(lmp_bin)
    if wdir is not None:
        wdir = Path(wdir)
    if logfile is not None:
        logfile = Path(logfile)
    if trajfile is not None:
        trajfile = Path(trajfile)
    
    if mult_mat is None:
        mult_mat = np.diag([1, 1, 1])
    
    if ismpi == False:
        nproc = 1
        mpirun = ''
    
    if temperature is None:
        raise TypeError('Please specify a temperature!')
    elif ucell_path is None:
        raise TypeError('Please specify a string for ucell_path!')
    elif pair_style is None:
        raise TypeError('Please specify pair_style for lammps!')
    elif pair_coeff is None:
        raise TypeError('Please specify a pair_coeff for lammps!')
    elif lmp_bin is None:
        if ismpi == True:
            lmp_bin = Path('lmp_mpi')
        else:
            lmp_bin = Path('lmp')
    elif ase_lammps_command is None:
        ase_lammpsrun_command = f'{mpirun} -n {nproc} {lmp_bin.absolute()}'

    
        ############################################################################### 
        
    def run_dynamics(atoms, dt, nsteps, loginterval, isnpt=False, iso=None):
        '''
        Function to run the MD (either NPT or NVT)
        '''
        if isnpt:
            press = 0
            workdir = wdir.joinpath('NPT')
            if iso == None:
                iso = 'iso'
            elif iso == True:
                iso = 'iso'
            elif iso == False:
                iso = 'aniso'
            else:
                raise ValueError('Since isnpt = True, iso must be either True, False or None (= True by default); any other value is unacceptable.')
        else:
            press = None
            workdir = wdir.joinpath('NVT')
        ## DEBUG
        if not isnpt:
            write(wdir.joinpath('save_atoms.traj'), atoms)
        ####
        state = LammpsState(temperature,
                            press,
                            dt = dt,
                            nsteps = nsteps,
                            loginterval = loginterval,
                            logfile = 'mlmd.log',
                            trajfile = 'mlmd.traj',
                            ptype=iso,
                            workdir = workdir.absolute())
        state.initialize_momenta(atoms) 

        # RUN THE MD
        #os.environ['ASE_LAMMPSRUN_COMMAND'] = ase_lammpsrun_command

        if isnpt and not wdir.joinpath(f'T{temperature}K_unitcell.json').exists():
            state.run_dynamics(atoms, pair_style, pair_coeff)
        if not isnpt and not wdir.joinpath(f'T{temperature}K.traj').exists():
            state.run_dynamics(atoms, pair_style, pair_coeff)


        # COLLECT THE RESULTS
        # Now the MD has run, so we can store the resulting configurations inside traj
        traj = read(workdir.joinpath('Trajectory/mlmd.traj').absolute(), index=f'{nthrow}:')
        
        # if it is NPT and the thermalised unitcell is not already there, then we want to create the thermalised unit cell
        # with the data we collected with the MD run
        if isnpt and not wdir.joinpath(f'T{temperature}K_unitcell.json').exists():
            cell = np.array([at.get_cell().array for at in traj]) # create an array of supercells
            cell = cell.mean(axis=0) # make the average supercell
            cell = invmult @ cell # reduce the average supercell to the average unitcell
            new_ucell = ucell.copy()
            new_ucell.set_cell(cell, True)
            write(wdir.joinpath(f'T{temperature}K_unitcell.json').absolute(), new_ucell) # write the cell into a file

        # if it is NVT and the trajectory is not there, then let us create a calculator and a new trajectory
        # file in which we can store all the configurations we got wit the MD along with their energy computes with the
        # calculator.
        elif not wdir.joinpath(f'T{temperature}K.traj').exists():
            newtraj = reconstruct_mlmd_trajectory(workdir.joinpath('Trajectory/mlmd.traj').absolute(), workdir.joinpath('Trajectory/mlmd.log').absolute())[nthrow:]
            write(wdir.joinpath(f'T{temperature}K.traj').absolute(), newtraj)
            #traj = reconstruct_mlmd_trajectory(f'{workdir}mlmd.traj', f'{workdir}mlmd.log')[nthrow:]
            #os.environ["ASE_LAMMPSRUN_COMMAND"] = f"{mpirun} -n {nproc} {lmp_bin.absolute()}"
            #calc = LAMMPS(pair_style=pair_style, 
            #              pair_coeff=pair_coeff,
            #              tmp_dir=Path('WTFASE').absolute())
            #for at in traj:
            #    at.calc = calc
            #    at.get_potential_energy()
            #    newtraj.write(at)
        else:
            pass


    
    
        ###############################################################################               
    
    
    
    ucell = read(ucell_path.absolute())
    invmult = np.linalg.inv(mult_mat) # this is used after NVT to reduce the thermalised supercell and obtain the thermalised UNIT cell

    os.environ['ASE_LAMMPSRUN_COMMAND'] = ase_lammpsrun_command
    if make_wdir == True:
        wdir.mkdir(parents=True, exist_ok=True)
    else:
        if not wdir.exists():
            print(f'The working directory {wdir.absolute()} does not exist! Either create it or set make_wdir = True')
            exit()
            
    if NPT == True:
        #first run NPT to thermalise
        atoms = make_supercell(ucell, mult_mat)
        run_dynamics(atoms = atoms,
                     dt = dt,
                     nsteps = nsteps,
                     loginterval = loginterval,
                     isnpt=True,
                     iso=iso)

    if NVT == True:
        if nvt_scell == None:
            # read the thermalised unitcell
            nvt_scell = read(wdir.joinpath(f'T{temperature}K_unitcell.json').absolute())
            # run the NVT with the thermalised unitcell
            nvt_scell = make_supercell(nvt_scell, mult_mat)
        
        
        run_dynamics(atoms = nvt_scell,
                     dt = dt,
                     nsteps = nsteps,
                     loginterval = loginterval,
                     isnpt=False)

        
        
def make_md(mode='interactive', fpath=None):
    '''Function to set the folders and scripts for the usual MD (NPT + NVT) run
    
    Parameters
    ----------
    
    mode: {'interactive', 'from_file'}
        - interactive: the input will be asked to the user
        - from_file: the input will be extracted from a file
    fpath: str
        path to the file containing the instructions
        
    '''

    parser = argparse.ArgumentParser(description="make_md script!")
    
    parser.add_argument("--mode", type=str, default='interactive', help="Mode of using this tool:\n\t- interactive: the input will be asked to the user\n\t- from_file: the input will be extracted from a file")
    parser.add_argument("--fpath", type=str, default=None, help="path to the file containing the instructions")
    
    args = parser.parse_args()
    
    mode = args.mode
    if mode not in ['interactive', 'from_file']:
        raise ValueError('The parameter "mode" must be either "interactive" or "from_file"')
    if mode == 'from_file':
        fpath = args.fpath
        if fpath == None:
            raise ValueError('When "mode" = "from_file" a file path must be passed as "fpath"')
        else:
            fpath = Path(fpath)

        
    ######## FUNCTIONS ########
    
    def ask_input():
        input_pars = dict()
        input_pars['wdir'] = input("Where?\n")
        time = []
        time.append(input("How much time to request for NPT? (HH:MM:SS)\n"))
        ncores = []
        ncores.append(input("How many cores for NPT?\n"))
        nsteps = []
        nsteps.append(input("How many timesteps for NPT?\n")) 
        nsteps.append(input("How many timesteps for NVT?\n"))
        loginterval = []
        loginterval.append(input("Loginterval for NPT?\n"))
        nthrow = []
        nthrow.append(input("How many initial timesteps NOT to include in the trajectory for NPT (ntrhow)?\n"))   
        nthrow.append(input("How many initial timesteps NOT to include in the trajectory for NVT? (nthrow)\n"))
        input_pars['ninstances'] = input("How many NPT instances?\n")   
        dt = []
        dt.append(input("Duration of a timestep (in fs) for NPT:\n"))     
        bool_nvt = input("Do you want the NVT MD runs to have the same setting (apart from nthrow, and nsteps, already asked separately) (yes/no)\n")    
        if bool_nvt == 'yes':
            time.append(time[0])
            ncores.append(ncores[0])       
            loginterval.append(loginterval[0])      
            dt.append(dt[0])   
        else:
            time.append(input("How much time to request for NVT? (HH:MM:SS)\n"))       
            ncores.append(input("How many cores for NVT?\n"))       
            loginterval.append(input("Loginterval for NVT?\n"))
            dt.append(input("Duration of a timestep (in fs) for NPT:\n"))

        bool_iso = input("Should isotropy be enforced on the system during NPT (yes/no)?\n")
        if bool_iso == 'yes':
            input_pars['iso'] = True      
        else:
            input_pars['iso'] = False

        bool_refine = input("Should the average NPT cell be refined before being used in the NVT (yes/no)?\n")
        if bool_refine == 'yes':
            input_pars['refine'] = True       
        else:
            input_pars['refine'] = False

        input_pars['time'] = time
        input_pars['ncores'] = ncores
        input_pars['nsteps'] = nsteps
        input_pars['loginterval'] = loginterval
        input_pars['nthrow'] = nthrow
        input_pars['dt'] = dt
        
        input_pars['matrix'] = input("Give the multiplication matrix for the supecell. Give all the nine elements 11, 12, 13, 21, 22, 23, 31, 32, 33\n")
        input_pars['matrix'] = [str(x) for x in input_pars['matrix'].split()]
        input_pars['init_structure'] = [str(x) for x in input('Give the path to the initial structure (must be a POSCAR file)').split()]
        input_pars['lammps_bin'] = [str(x) for x in input('Give the path to the lammps binary\n').split()]
        input_pars['mpirun'] = [" ".join([str(x) for x in input('Give any instruction to write before the call to lammps and mpt (e.g. mpirun)\n').split()])]
        input_pars['pair_style'] = [str(x) for x in input('Give the lammps pair style\n').split()]
        
        mode = input("Choose mode\n1 - Homogeneous custom range\n2 - Custom list of temperatures\n")

        if str(mode) == '1':
            input_pars['mode'] = ['1']
            input_pars['T1'] = input("T start (K); integer:\n")
            input_pars['T2'] = input("T stop (K); integer (excluded):\n")
            input_pars['step'] = input("step (K); integer:\n")
        elif str(mode) == '2':
            input_pars['mode'] = ['2']
            ntemps = input("How many temperatures?\n")
            temps = []
            for i in range(int(ntemps)):
                temp = input(f"Give the temperature n. {i+1}\n")
                temps.append(temp)
            input_pars['temps'] = temps
        else:
            print("You are not the smart one in your  family, are you?")
            
        # let's convert the string into list of tokens
#         print(list(input_pars.values()))
#         for i in range(len(list(input_pars.keys()))):
#             input_pars[list(input_pars.keys())[i]] = list(input_pars.values())[i].split()
            
        input_pars = convert_input_pars(input_pars)
        
        return input_pars

    def get_input_from_file(filepath):
        filepath = Path(filepath)
        with open(filepath, 'r') as fl:
            lines = fl.readlines()
        input_pars = dict()
        for line in lines:
            if line.split() == []:
                continue
            var_name = line.split()[0]
            var_content = " ".join([str(x) for x in line.split()[1:]]).split()
            input_pars[var_name] = var_content
        input_pars['iso'] = eval(input_pars['iso'][0])
        input_pars['refine'] = eval(input_pars['refine'][0])
        input_pars['parallel'] = eval(input_pars['parallel'][0])
        input_pars = convert_input_pars(input_pars)
        
        return input_pars

    def convert_input_pars(input_pars):
        input_pars['wdir'] = Path(input_pars['wdir'][0])
        input_pars['mpirun'] = str(input_pars['mpirun'][0])
        input_pars['lammps_bin'] = Path(input_pars['lammps_bin'][0])
        if len(input_pars['pair_style']) == 2:
            if Path(input_pars['pair_style'][1]).absolute().is_file() == True:
                pp = Path(input_pars['pair_style'][1]).absolute()
            else:
                pp = input_pars['pair_style'][1]
            input_pars['pair_style'] = " ".join([input_pars['pair_style'][0], str(pp)])
        else:
            input_pars['pair_style'] = str(input_pars['pair_style'][0])
        input_pars['init_structure'] = Path(input_pars['init_structure'][0])
        input_pars['time'] = [str(x) for x in input_pars['time']] 
        input_pars['ncores'] = [int(x) for x in input_pars['ncores']]
        input_pars['nsteps'] = [int(x) for x in input_pars['nsteps']]
        input_pars['loginterval'] = [int(x) for x in input_pars['loginterval']]
        input_pars['nthrow'] = [int(x) for x in input_pars['nthrow']]
        input_pars['ninstances'] = int(input_pars['ninstances'][0])
        input_pars['dt'] = [float(x) for x in input_pars['dt']]
        mat_elems = input_pars['matrix']
        matrix = [[mat_elems[0], mat_elems[1], mat_elems[2]],
                  [mat_elems[3], mat_elems[4], mat_elems[5]],
                  [mat_elems[6], mat_elems[7], mat_elems[8]]]
        input_pars['matrix'] = np.array(matrix, dtype='float')
        if 'mode' in input_pars.keys():
            input_pars['mode'] = int(input_pars['mode'][0])
            if input_pars['mode'] == 1:
                input_pars['temps'] = np.array([float(input_pars['T1']), float(input_pars['T2'])])
            else:
                input_pars['temps'] = np.array(input_pars['temps'], dtype='float')                
        else:
            input_pars['temps'] = np.array(input_pars['temps'], dtype='float')
        new_temps = []
        for temp in input_pars['temps']:
            if temp.is_integer():
                new_temps.append(int(temp))
            else:
                new_temps.append(temp)
        input_pars['temps'] = new_temps
                
        keys = list(input_pars.keys())
        values = list(input_pars.values())
        for i in range(len(values)):
            print(f'{keys[i]} ---> {values[i]}')
            
        return input_pars

    ################
    
    if mode == 'interactive':
        input_pars = ask_input()
    elif mode == 'from_file':
        input_pars = get_input_from_file(fpath.absolute())
        
    if input_pars['ncores'][0] > 1 or input_pars['ncores'][0] > 1:
        input_pars['parallel'] = True
    else:
        input_pars['parallel'] = False
    
    root_dir = input_pars['wdir']
    scripts_dir = root_dir.joinpath('scripts_md')
    scripts_dir.mkdir(parents=True, exist_ok=True)
    
    scripts_to_copy_dir = Path(__file__).parent.joinpath('data/make_md')
    
    file_to_copy_names = ['RunNPT_instance.py', 'RunNVT.py', 'LaunchNPT_instances.py', 'nvt_job.sh', 'npt_job.sh']
    
    for f in file_to_copy_names:
        os.system(f"cp {scripts_to_copy_dir.joinpath(f).absolute()} {scripts_dir.joinpath(f).absolute()}")
    
    if input_pars['init_structure'].is_file() == True:
        os.system(f"cp {input_pars['init_structure'].absolute()} {root_dir.joinpath('unitcell.poscar').absolute()}")
    else:
        raise FileNotFound(f'The initial structure was not found!')
        
    RunNPT_path = scripts_dir.joinpath('RunNPT_instance.py')
    RunNVT_path = scripts_dir.joinpath('RunNVT.py')
        
    matrix = input_pars['matrix']
    
    for i, filepath in enumerate([RunNPT_path, RunNVT_path]):
        newlines = []
        tokens = np.zeros(20)
        with open(filepath.absolute(), 'r') as fl:
            lines = fl.readlines()
        for j, line in enumerate(lines):
            if 'root_dir =' in line and tokens[0] == 0:
                newlines.append(f'root_dir = \'{root_dir.absolute()}\'\n')
                tokens[0] = 1
            elif 'nproc' in line and tokens[1] == 0:
                newlines.append(f"nproc = {input_pars['ncores'][i]}\n")
                tokens[1] += 1
            elif 'nsteps' in line and tokens[2] == 0:
                newlines.append(f"nsteps = {input_pars['nsteps'][i]}\n")
                tokens[2] += 1
            elif 'loginterval' in line and tokens[3] == 0:
                newlines.append(f"loginterval = {input_pars['loginterval'][i]}\n")
                tokens[3] += 1
            elif 'nthrow' in line and tokens[4] == 0:
                newlines.append(f"nthrow = {input_pars['nthrow'][i]}\n")
                tokens[4] += 1
            elif 'dt' in line and tokens[5] == 0:
                newlines.append(f"dt = {input_pars['dt'][i]}\n")
                tokens[5] += 1
            elif 'iso' in line and tokens[6] == 0:
                newlines.append(f"iso = {input_pars['iso']}\n")
                tokens[6] += 1
            elif 'mult_mat' in line and tokens[7] == 0:
                newlines.append(f"mult_mat = np.array([[{matrix[0,0]}, {matrix[0,1]}, {matrix[0,2]}], [{matrix[1,0]}, {matrix[1,1]}, {matrix[1,2]}], [{matrix[2,0]}, {matrix[2,1]}, {matrix[2,2]}]])\n")
                tokens[7] += 1
            elif 'refine' in line and tokens[8] == 0:
                newlines.append(f"refine = {str(input_pars['refine'])}\n")
                tokens[8] += 1
            elif 'mpirun' in line and tokens[9] == 0:
                newlines.append(f"mpirun = \'{input_pars['mpirun']}\'\n")
                tokens[9] += 1
            elif 'pair_style' in line and tokens[10] == 0:
                newlines.append(f"pair_style = \'{input_pars['pair_style']}\'\n")
                tokens[10] += 1
            elif 'lmp_bin' in line and tokens[11] == 0:
                newlines.append(f"lmp_bin = \'{input_pars['lammps_bin'].absolute()}\'\n")
                tokens[11] += 1
            elif 'ismpi' in line and tokens[12] == 0:
                newlines.append(f"ismpi = {input_pars['parallel']}\n")
                tokens[12] += 1
            else:
                newlines.append(line)
        with open(filepath.absolute(), 'w') as fl:
            fl.writelines(newlines)
    
    npt_job_path = scripts_dir.joinpath('npt_job.sh')
    nvt_job_path = scripts_dir.joinpath('nvt_job.sh')
    
    for i, filepath in enumerate([npt_job_path, nvt_job_path]):
        newlines = []
        with open(filepath.absolute(), 'r') as fl:
            lines = fl.readlines()
        for line in lines:
            if '#SBATCH --ntasks=' in line:
                newlines.append(f"#SBATCH --ntasks={input_pars['ncores'][i]}\n")
            elif '#SBATCH --time=' in line:
                newlines.append(f"#SBATCH --time={input_pars['time'][i]}\n")
            else:
                newlines.append(line)
        with open(filepath.absolute(), 'w') as fl:
            fl.writelines(newlines)

    LaunchNPT_instances_path = scripts_dir.joinpath('LaunchNPT_instances.py')
    newlines = []
    with open(LaunchNPT_instances_path.absolute(), 'r') as fl:
        lines = fl.readlines()
    for line in lines:
        if 'root_dir =' in line:
            newlines.append(f"root_dir = \'{root_dir.absolute()}\'\n")
        elif 'ninstances =' in line:
            newlines.append(f"ninstances = {input_pars['ninstances']}\n")
        else:
            newlines.append(line)
    with open(LaunchNPT_instances_path.absolute(), 'w') as fl:
        fl.writelines(newlines)


    for T in input_pars['temps']:
        temp_dir = root_dir.joinpath(f'T{T}K/')
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.system(f"ln -s -f {scripts_dir.absolute().joinpath('*')} {temp_dir.absolute()}")
        os.system(f"ln -s -f {root_dir.absolute().joinpath('unitcell.poscar')} {temp_dir.absolute()}")
    
            

def time_convergence_npt(traj, mult_mat=np.eye(3), units=['$\AA$', 'fs']):
    '''Given an NPT trajectory, the convergence of the thermalized unitcell parameters with time is checked
    
    Prameters
    ---------
    traj: ASE.Atoms object
        Trajectory to analyze
    mult_mat: numpy array; default= identity
        3x3 matrix used to go from the unitcell to the supercell; the trajectory is supposed to contain 
        the supercells and the inverse of mult_mat will be used to get back to the unitcell (U = P^(-1)S)
    units: list of two strings; default= ['$\AA$', 'fs']
        units of the parameters and time
    '''
    
    n_structs = len(traj)
    cells = np.zeros((n_structs, 3, 3))
    for i in range(n_structs):
        cells[i] += np.array([x.get_cell() for x in traj[:i+1]]).mean(axis=0)
    B = np.linalg.inv(mult_mat)
    ucells = np.array([make_supercell(x, B) for x in cells])

    a = np.linalg.norm(cells[:, 0], axis=1)
    b = np.linalg.norm(cells[:, 1], axis=1)
    c = np.linalg.norm(cells[:, 2], axis=1)
    # PLOT
    
    # a, b, c, a/b, a/c and b/c 
    
    fig, ax1 = plt.subplots(3, 2, figsize=(10, 15) )
    
    ax1[0].plot(a)
    ax1[0].ylabel(f'a ({units[0]})')
    ax1[0].xlabel(f'Time ({units[1]})')
    
    ax1[1].plot(b)
    ax1[1].ylabel(f'b ({units[0]})')
    ax1[1].xlabel(f'Time ({units[1]})')
    
    ax1[2].plot(c)
    ax1[2].ylabel(f'c ({units[0]})')
    ax1[2].xlabel(f'Time ({units[1]})')
    
    ax1[3].plot(a/b)
    ax1[3].ylabel(f'a/b ')
    ax1[3].xlabel(f'Time ({units[1]})')
    
    ax1[4].plot(a/c)
    ax1[4].ylabel(f'a/c ')
    ax1[4].xlabel(f'Time ({units[1]})')
    
    ax1[5].plot(b/c)
    ax1[5].ylabel(f'b/c ')
    ax1[5].xlabel(f'Time ({units[1]})')
    
    fig.set_suptitle('Cell parameters for the thermalized unitcell vs. time of simulation')
    

def wrap(*args):
    def modified_decimal_part(arr):
        decimal_part = np.abs(arr) - np.floor(np.abs(arr))  # Compute decimal part
        return np.where(arr >= 0, decimal_part, 1 - decimal_part)  # Apply transformation
    
    if len(sys.argv) < 2:
        fpath = 'Trajectory.traj'
    else:
        fpath = sys.argv[1] 
    fpath = Path(fpath)
    ats = read(fpath, index=':')
    new_ats = []
    for at in ats:
        cell = at.get_cell()
        pos = at.get_scaled_positions()
        new_pos = modified_decimal_part(pos)    
        at.set_scaled_positions(new_pos)
    parent = fpath.parent
    fname = 'w_' + fpath.name
    write(parent.joinpath(fname), ats)
    

def rdf(cell, pos, centers, coords, V=None, nbins=100, cutoff=None):   
    '''Function to calculate the radial distribution function from a set of positions. The cutoff will be chosen 
    automatically as half of the minimum axis lentgh (or the maximum distance between atoms, if it is shorter)/ 

    Parameters
    ----------
    cell: np.array (3x3)
        axis of the box. cell[i,j] = j-th cartesian component of the i-th axis
    pos: np.array (Nx3)
        positions of the particles in cartesian coordinates; N = number of particles. 
        pos[i,j] = j-th cartesian component of the position of the i-th particle
    centers: np.array(Mx3)
        List of the indices of the atoms (centers) to compute the rdf for; centers[i] = index of the position of the i-th center in
        pos.
    coords: list
        List of the indices of the atoms (coords) to consider for each center; coords[i,j] = index of the position of the j-th coord 
        for the i-th center in pos. Note: if two or more centers have a different number of coords, the function will slow down!
    V: float; default=np.linalg.det(cell)
        volume of the box to use to renormalize the rdf
    nbins: int; default=100
        number of x-values of the rdf. Beware: too big nbins will lead to errors!
    cutoff: float; default= the minimum distance between the center of the cell and the surface. 
    Returns
    -------
    rdf_at: np.array(Nxnbins)
        radial distribution functions for each center, considering the respective coords
    bincs: np.array(nbins)
        x-values of the rdf
    cutoff: float
        cutoff used for the rdf (it should close to the last element of bincs)
    '''

    
    
    # type checks
    if not isinstance(cell, type(np.array([]))):
        raise TypeError('cell must be a 3x3 numpy array!')
    elif not cell.shape == np.ones((3,3)).shape:
        raise TypeError('cell must be a 3x3 numpy array!')
    elif np.linalg.det(cell) == 0:
        raise ValueError('the cell axes are not linearly indendent!')

    if not isinstance(pos, type(np.array([]))):
        raise TypeError('pos must be a numpy array!')
    elif len(pos.shape) != 2 or pos.shape[1] != 3:
        raise TypeError('pos elements must be 1x3 numpy arrays!')

    if not isinstance(centers, type(np.array([]))):
        raise TypeError('centers must be a numpy array!')

    if not isinstance(coords, list) or not all([isinstance(x, list) for x in coords]):
        raise TypeError('coords must be a list of lists!')
    elif not all([isinstance(x, int) for y in coords for x in y]):
        raise TypeError('coords must be a list of lists of integers!')

        
    
    # we need the reduced coordinates
    rpos = np.linalg.inv(cell.T) @ pos.T
    rpos = rpos.T # Nx3
    
    if V is None:
        V = abs(np.linalg.det(cell))
    
    # let's define the maximum cutoff (it will be then compared to the maximum atomic distance)
    max_cutoff = min_distance_to_surface(cell)
    
    cell_n = np.linalg.norm(cell, axis=1) # auxiliary variable with the lenghts of the axes
    
    ncenters = len(centers)
    ncoords = np.array([len(x) for x in coords])

    if (ncoords == ncoords[0]).all():

        # do you want to understand the following expression? Go to the end of the function, then.
        D = np.linalg.norm(np.transpose(cell @ np.transpose(mic_sign(rpos[centers, None, :] - rpos[coords]), (0, -1, -2)), (0, -1, -2)), axis=2)
        #D = np.linalg.norm(mic(rpos[:, None, :] - rpos[coords]) @ cell, axis=2)
        
        # let's get the density
        rho = ncoords[0] / V # for each center, the number of coordinants is the same, so this formula holds for any center
    
        # BINNING
        # let's create the bin centers
        cutoff = max_cutoff * 1.0001
        #cutoff = min(D.max(), max_cutoff) * 1.00001 # Ensure cutoff doesn't exceed half the box size; the scaling is
                                                    # needed for numerical reasons
        step = cutoff / nbins
        bincs = np.array(range(0,nbins)) * step + step/2 
        
        # let's create the bin volumes
        binvs = (4/3) * np.pi * ((bincs + step/2)**3 - (bincs - step/2)**3)
    
        rdf_at = np.zeros((ncenters, len(bincs))) # initialize the rdf for all the centers
        for i in range(D.shape[0]): # loop over the atoms
            # let's turn each element of the row into the respective nearest element in the bin centers
            indices = np.floor((np.extract(D[i]<cutoff, D[i]) - step/2) / step + 0.5).astype(int)
            approx = bincs[indices]
            for j in range(len(bincs)):
                rdf_at[i,j] += (indices == j).sum()
        # now rdf_at contains the rdf for each atom
        rdf_at /= binvs * rho
    
        #return rdf_at, bincs, cutoff
        
    else:
        # first create a set of unique number of coords
        set_nums = list(set([len(x) for x in coords]))
        set_nums.sort()
        center_groups = []
        for set_num in set_nums:
            center_group = []
            for i, ncoord in enumerate(ncoords):
                #print(f'ncoord= {ncoord}, set_num= {set_num}')
                if ncoord == set_num:
                    center_group.append(i)
            center_group = np.array(center_group, dtype='int')
            center_groups.append(center_group)

        rdf_collection = []
        for center_group in center_groups:
            res, bincs, cutoff = rdf(cell, pos, centers[center_group], [coords[i] for i in center_group], V=None, nbins=100)
            rdf_collection.append(res)
        rdf_collection = np.vstack(rdf_collection)   
        flat_center_groups = np.concatenate(center_groups)  # now here we have the indices of the centers in the same order as in rdf_collection
        sort_indices = np.argsort(flat_center_groups)
        rdf_at = rdf_collection[sort_indices]
            # bincs and cutoff that will be returned are those from the last iteration of the for loop
        
    # Before returning let's break down the hell broadcasting used to generate D
    # Let's start by taking rpos[coords]: it has shape (ncenters, ncoords, 3) and for each center there is a row for each 
    # coordinating atom (coord) and for each coord there are three reduced coordinates (hence a vector). 
    # We need to subtract each vector associated to a single center to that center's rpos. So we take rpos[centers], which has shape
    # (ncenters,3) and subtract to it rpos[coords]. In terms of shapes: (ncenters,3) - (ncenters, ncoords, 3). Following the broadcasting
    # rules of numpy, the last two dimensions are mapped between the two arrays, while for the third one (from right), a 1 
    # is assumed for the first shape. The operation now would be: (1,ncenters,3) - (ncenters,ncoords,3). The dimensions do not correspond.
    # In order to solve this, we need to change the shape from (1, ncenters, 3) to (ncenters, 1, 3). This way, in the second dimension
    # the elements are duplicated as many times as needed to perform the subtraction. Namely, rpos[centers][i,j] = rpos[centers][i,j+1]...
    # To do so, we need to add a dimension in the middle of rpos[centers] ---> rpos[centers][:, None, :]. Now this has a shape that will be
    # correctly broadcasted. 
    # Now that the operation rpos[centers][:, None, :] - rpos[coords] has the proper shape: (ncenters, 1, 3) - (ncenters, ncoords, 3),
    # we obtain an array of distances between the centers and their respective coords, having shape (ncenters, ncoords, 3).
    # We need to apply the mininum image convention to them, hence we pass them to the mic() function (it's simple enough to not deserve
    # an explanation). So far the instruction is:
    #     mic(rpos[centers][:, None, :] - rpos[coords])
    # This gives us a similar array with the same shape (ncenters, ncoords, 3), the only difference is in the actual numbers inside.
    # Now we need to convert them to cartesian coordinates (before getting the norm).
    # The matrix formula to convert a vector from the reduced basis to the cartesian one is x_c = cell @ x_r, where cell[i,j] is the
    # i-th cartesian component of the j-th unit vector, and both x_c and x_r are column vectors, hance have shape (3,1).
    # In our case both cell and the vectors are transposed, so we need*** to transpose both before multiplying, multiply them and then
    # re-transpose the result. To transpose the cell we just write cell.T, to transpose the distance vectors, we need to use the np.transpose
    # function, as the vectors are themselved inside arrays. Remember that the distance array now has shape (ncenters, ncoords, 3);
    # we need to transpose each matrix defined by the last two indices, hence we have to swap the last two dimensions.
    # We do it by using np.transpose(distance_vector, (0, -1, -2)).
    # The instruction so far is:
    #     cell.T @ np.transpose(mic(rpos[centers][:, None, :] - rpos[coords]), (0, -1, -2))
    # The result has the shape (ncenters, 3, ncoords), and we need to re-transpose it to take it back to the previous form:
    #     np.transpose(cell.T @ np.transpose(mic(rpos[centers][:, None, :] - rpos[coords]), (0, -1, -2)), (0, -1, 2))
    # Now we have a (ncenters, ncoords, 3) array with the distance vectors in cartesian coordinates and we just need to get the norm.
    #     np.linalg.norm(np.transpose(cell.T @ np.transpose(mic(rpos[centers][:, None, :] - rpos[coords]), (0, -1, -2)), (0, -1, 2)), axis=2)
    # where we asked to compute the norm on the last axis.
    # That's it!

    # ***In principle we can avoid the transposition, indeed since both the cell and the vectors are transposed, we can just invert the order
    # in the multiplication. In this case the final instruction would be shorter and simpler:
    #     np.linalg.norm(mic(rpos[:, None, :] - rpos[coords]) @ cell, axis=2)
    # however, for some reason it is slower.
    
    return rdf_at, bincs, cutoff
    
    
    
    
    
