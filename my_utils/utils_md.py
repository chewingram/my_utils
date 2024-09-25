import os
import numpy as np
from ase.io import read, write, Trajectory
from ase.build import make_supercell
from ase.calculators.lammpsrun import LAMMPS
from mlacs.state import LammpsState
from mlacs.utilities.io_lammps import reconstruct_mlmd_trajectory
import sys
from .utils import path
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
                                  
    log_file(str): name of the log file; it will be saved in the lammps workdir (NPT or NVT)
    
    traj_file(str): name of the trajectory file; it will be saved in the lammps workdir (NPT or NVT).
    
    iso(bool): mandatory when npt=True, useless otherwise. If True, then the NPT MD is done enforcing isotropy; if False, then anisotropy.
    '''
    
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
            lmp_bin = f'lmp_mpi'
        else:
            lmp_bin = f'lmp'
    elif ase_lammps_command is None:
        ase_lammpsrun_command = f'{mpirun} -n {nproc} {lmp_bin}'
            
    wdir = path(wdir)
    
        ############################################################################### 
        
    def run_dynamics(atoms, dt, nsteps, loginterval, isnpt=False, iso=None):
        '''
        Function to run the MD (either NPT or NVT)
        '''
        if isnpt:
            press = 0
            workdir = f'{wdir}NPT/'
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
            workdir = f'{wdir}NVT/'

        state = LammpsState(temperature,
                            press,
                            dt = dt,
                            nsteps = nsteps,
                            loginterval = loginterval,
                            logfile = 'mlmd.log',
                            trajfile = 'mlmd.traj',
                            ptype=iso,
                            workdir = workdir)
        state.initialize_momenta(atoms) 

        # RUN THE MD
        #os.environ['ASE_LAMMPSRUN_COMMAND'] = ase_lammpsrun_command

        if isnpt and not os.path.exists(f'{wdir}T{temperature}K_unitcell.json'):
            state.run_dynamics(atoms, pair_style, pair_coeff)
        if not isnpt and not os.path.exists(f'{wdir}T{temperature}K.traj'):
            state.run_dynamics(atoms, pair_style, pair_coeff)


        # COLLECT THE RESULTS
        # Now the MD has run, so we can store the resulting configurations inside traj
        traj = read(f'{workdir}mlmd.traj', index=f'{nthrow}:')

        # if it is NPT and the thermalised unitcell is not already there, then we want to create the thermalised unit cell
        # with the data we collected with the MD run
        if isnpt and not os.path.exists(f'{wdir}T{temperature}K_unitcell.json'):
            cell = np.array([at.get_cell().array for at in traj]) # create an array of supercells
            cell = cell.mean(axis=0) # make the average supercell
            cell = invmult @ cell # reduce the average supercell to the average unitcell
            new_ucell = ucell.copy()
            new_ucell.set_cell(cell, True)
            write(f'{wdir}T{temperature}K_unitcell.json', new_ucell) # write the cell into a file

        # if it is NVT and the thermalised supercell is not there, then let us create a calculator and a new trajectory
        # file in which we can store all the configurations we got wit the MD along with their energy computes with the
        # calculator.
        elif not os.path.exists(f'{wdir}T{temperature}K.traj'):
            newtraj = Trajectory(f'{wdir}T{temperature}K.traj', 'w')
            #traj = reconstruct_mlmd_trajectory(f'{workdir}mlmd.traj', f'{workdir}mlmd.log')[nthrow:]
            os.environ["ASE_LAMMPSRUN_COMMAND"] = f"mpirun -n {nproc} {lmp_bin}"
            calc = LAMMPS(pair_style=pair_style, pair_coeff=pair_coeff,
                      tmp_dir="WTFASE")
            for at in traj:
                at.calc = calc
                at.get_potential_energy()
                newtraj.write(at)


    
    
        ###############################################################################               
    
    
    
    ucell = read(ucell_path)
    invmult = np.linalg.inv(mult_mat) # this is used after NVT to reduce the thermalised supercell and obtain the thermalised UNIT cell

    os.environ['ASE_LAMMPSRUN_COMMAND'] = ase_lammpsrun_command
    if make_wdir == True:
        os.system(f'mkdir -p {wdir}')
    else:
        if not os.exists(wdir):
            print(f'The working directory {wdir} does not exist! Either create it or set make_wdir = True')
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
            nvt_scell = read(f'{wdir}T{temperature}K_unitcell.json')
            # run the NVT with the thermalised unitcell
            nvt_scell = make_supercell(nvt_scell, mult_mat)
        
        
        run_dynamics(atoms = nvt_scell,
                     dt = dt,
                     nsteps = nsteps,
                     loginterval = loginterval,
                     isnpt=False)

       

            

