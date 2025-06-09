import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle as pkl
from typing import List, Union
from copy import deepcopy as cp
from ase.atoms import Atoms
from ase.io import read, write
from ase.build import make_supercell as mk_supercell
import os
import sys
from . import utils_tdep as tdp
from . import utils_mlip as mlp
from .utils import ln_s_f, min_distance_to_surface
from subprocess import run
import shutil
from ase.calculators.singlepoint import SinglePointCalculator
import h5py



p = Path(shutil.which('extract_forceconstants')).parent
if p != None:
    tdp_bin_dir = p
else:
    tdp_bin_dir = Path('./')
#print(f'tdp bin path: {tdp_bin_dir.absolute()}')
    
class logger():
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if os.path.exists(filepath):
            filepath.unlink(missing_ok=True)
    
    def log(self, text):
        text = f'{text}'
        print(text)
        with open(self.filepath.absolute(), 'a') as fl:
            fl.write(f'{text}\n')
            
    
    
def compute_props_MTP(dir, atoms=None, mlp_bin=None, mlp_pot=None, mpirun='mpirun'):
    '''
    Function to compute properties with MTP. It also save the struct + props as an ase traj called new_confs.traj
    Args:
    dir(str): directory where MTP will run
    atoms(ase trajectory): list of ase.Atoms objects to compute properties of
    mlp_bin(str): path to the binary of MTP
    mlp_pot(str): path to the trained potential (.mtp) file to use
    mpirun(str): command to write before the binary (e.g. 'srun'), default='mpirun'
    '''
    
    dir = Path(dir)
    if mlp_bin is not None:
        mlp_bin = Path(mlp_bin)
    if mlp_pot is not None:
        mlp_pot = Path(mlp_pot)
    
    atoms_with_props = mlp.calc_efs_from_ase(mlip_bin=mlp_bin.absolute(),
                                             atoms=cp(atoms), 
                                             mpirun=mpirun,
                                             pot_path=mlp_pot.absolute(), 
                                             cfg_files=False, 
                                             out_path='./out.cfg',
                                             dir=dir.absolute(), 
                                             write_conf=False,
                                             outconf_name=None)
    
    write(dir.joinpath('new_confs.traj').absolute(), atoms_with_props)



def compute_true_props(dir, atoms, confs_path, function_to_compute, params_to_compute):
    '''
    Function to compute energy, forces and stress for some configurations. Produces two files:
    1) new_confs.traj: ase trajectory containing struct + props of the confs in "atoms"
    2) "confs_path": an ase trajectory containing struct + propr of the old confs (if present) + new ones
    The file 1) is produced by another function which is called (function_to_compute), so this function actually
    merge the new confs + props to the old ones.
    Args:
    dir(str): directory where the calculation is run
    atoms(ase trajectory): list of ase.Atoms objects that normally still don't have properties
    confs_path(str): path to the file containing the pre-existing structs + props (it will be appended with
                     the new ones, so it can also not exist.)
    function_to_compute(funct): function to run the calculation of the properties
    params_to_compute(dict): dictionary with parameters needed by the calculating function (apart from "dir"
                             and "atoms": if ever needed, the are already given as args of this function).
    '''
    dir = Path(dir)
    confs_path = Path(confs_path)
    
    # compute props and save the new structs
    function_to_compute(dir.absolute(), atoms, **params_to_compute)
    # now new_confs.traj contains new_struct + props, we can merge with the existing ones
    new_ats = read(dir.joinpath('new_confs.traj'), index=':')
    ats_to_save = []
    if os.path.exists(confs_path.absolute()):
        existing_confs = read(confs_path.absolute(), index=':')
        ats_to_save.extend(existing_confs)
    ats_to_save.extend(new_ats)
    write(confs_path.absolute(), ats_to_save)

def msd_from_positions(confs, ref_conf):
    '''
    Function to compute the mean squared displacement from an ase trajectory (list of ase.Atoms objects).
    Beware! Currently this ONLY works for cubic supercells. To be updated using reduced coordinates and later
    converting into cartesian coordinates.
    Beware! This only works if the configurations all atoms are INSIDE the supercell (as with PBC). 
    Under this assumption two further assumptions need to be made:
        1. the particles never displace (along each direction) by a distance greater or equal to the cell axes.
        2. between two possible displacements (toward + or - of the axis) the particles displace the least distance
    Both assumptions are considered reasonable by assuming small displacements of the particles. Every displacement
    is defined w.r.t. a reference structure (e.g. equilibrium).
    Args:
    confs(list): list of ase.Atoms objects to compute the MSD for
    ref_conf(ase.Atoms): reference structure for the MSD
    Returns:
    msd_tot(list): list of msd along x, y, z and "absolute".
    '''
    ## MEAN SQUARE DISPLACEMENT FROM POSITIONS 

    nconfs = len(confs)
    natoms = len(confs[0])
    cells = np.array([x.get_cell() for x in confs])

    lengths = cells[:, np.arange(3), np.arange(3)]

    lengths = lengths[:, np.newaxis, :]

    positions = np.array([x.get_positions() for x in confs])

    ref_pos = np.array(ref_conf.get_positions())

    diffs = np.absolute(positions - ref_pos)

    diffs2 = np.where(diffs < lengths / 2, diffs, lengths - diffs) # shape (nconfs, natoms, 3)

    sd = diffs2 ** 2 # shape (nconfs, natoms, 3)

    sd_directions = sd.reshape(-1, sd.shape[-1]).transpose() # shape (3, nconfs*natoms)

    sd_vect = np.linalg.norm(diffs2.reshape(-1, sd.shape[-1]), axis=1).transpose() # shape (3, nconfs*natoms)

    sd_tot = np.vstack((sd_directions, sd_vect)) # shape (4, nconfs*natoms)

    msd_tot = sd_tot.mean(axis=1) # shape (4)

    return msd_tot

def old_run_stdep(root_dir=Path('./').absolute(),
              mpirun='',
              loto=False, 
              preexisting_ifc=True, 
              max_freq=None,
              T=0,
              quantum=True,
              rc2=5,
              rc3=5,
              first_order=False,
              max_iter=1000,
              max_confs=10000,
              nconfs_start=1,
              thr=0.00001,
              mlp_bin_path=None,
              mlp_pot_path = None,
              tdp_bin_dir = tdp_bin_dir):
    '''
    Function to run the s-dtep algorithm.
    Args:
    root_dir(str, path): directory where to run the algorithm (infiles ucell, scell, loto (if needed) and starting ifc 
                         (if needed) must already be there
    mpirun(str): command to add before the actual binaries
    loto(bool): True: the LO-TO splitting-correction is applied (infile.lotosplitting must be present); False: no LO-TO
    preexisting_ifc(bool): True: the pre-existing ifc are used (infile.forceconstant must be present); False: no pre-existing
                           ifc are used (then a maximum frequency must be specified, see "max_freq")
    max_freq(float): Maximum frequency (in THz) to use for the initial guess of the ifc in case no pre-existing ones are provided
    T(float): temperature in K
    quantum(bool): True: the Bose-Einstein distribution is used for phonons; False: classical distribution is used for phonons
    rc2(float): cutoff radius for 2nd-order ifc (in Angstrom)
    rc3(float): cutoff radius for 3nd-order ifc (in Angstrom)
    max_iter(int): maximum number of iterations (in each iteration usually many configurations are generated) 
    max_confs(int): maximum number of configurations to generate overall
    nconfs_start(int): currently, the minimum numer of configurations to generate at each iteration (except the first one).
                       The meaning of this variable is specific for the function to determine the number of confs. per each 
                       iteration. 
    thr(float): currently, the threshold for the RELATIVE (not %) error used to determine the convergence. The meaning of
                this variable is specific for the function used to determine the convergence
    mlp_bin_path(str, path): currently, the path to the MTP binary. The meaning of this variable is specific for the function 
                             used to compute the true properties.
    mlp_pot_path(str, path): currently, path to the trained MTP MLIP. The meaning of this variable is specific for the function
                             used to compute the true properties.
    tdp_bin_dir(str, path): path to the directory where all the tdep binaries are stored
                       
    '''
    
    ##### FUNCTIONS #####
    
    def path(path):
        path = os.path.abspath(path)
        if not path.endswith('/'):
            path = path + "/"
        return path

#     def check_convergence_msd(root_dir, thr):
#     TO BE REWRITTEN
#         #msds = msds[3]
#         if len(msds) < 2:
#             return [False, 0]
#         if msds[-2][1][3] == 0:
#             return [False, 0]
#         else:
#             err = (msds[-1][1][3] - msds[-2][1][3])/abs(msds[-2][1][3])
#         if abs(err) < thr:
#             return [True, err]
#         else:
#             return [False, err]
        

    
    def check_convergence(root_dir, conv_prop, confs_path, thr):  
        # actual function:
        root_dir = Path(root_dir)
        confs_path = Path(confs_path)
        ## TEMPORARY:
        nconfs = len(read(confs_path.absolute(), index=':'))
        conv_prop.append([nconfs, 0])
        if nconfs < 100000:
            return [False, 0]
        else:
            return [True, 0]
        #############
        #return check_convergence_free_energy(root_dir=root_dir, conv_prop=conv_prop, thr=thr)
    
    
    def check_convergence_free_energy(root_dir, conv_prop, thr):
        '''
        The anharmonic free energy will be computed up to the second order and the convergence checked with respect to the
        previous one.
        Args:
        root_dir(str, path): root directory where the directory 'ifc' is.
        '''
        root_dir = Path(root_dir)
        conv_dir = root_dir.joinpath('conv_free_energy')
        conv_dir.mkdir(parents=True, exist_ok=True)
        
        # first we need to copy the infiles
        ln_s_f({root_dir.joinpath("ifc/infile.stat")}, conv_dir)
        ln_s_f({root_dir.joinpath("ifc/infile.meta")}, conv_dir)
        ln_s_f({root_dir.joinpath("ifc/infile.positions")}, conv_dir)
        ln_s_f({root_dir.joinpath("ifc/infile.forces")}, conv_dir)
        ln_s_f({root_dir.joinpath("ifc/outfile.forceconstant")}, conv_dir.joinpath("infile.forceconstant"))
        ln_s_f({root_dir.joinpath("ifc/infile.ucposcar")}, conv_dir)
        ln_s_f({root_dir.joinpath("ifc/infile.ssposcar")}, conv_dir)
        
                  
        #if os.path.exists(f'{root_dir}ifc/outfile.forceconstant_thirdorder'):
        #    os.system(f'ln -s -f {root_dir}ifc/outfile.forceconstant_thirdorder {conv_dir}infile.forces')
        fe_params = dict(dir=conv_dir,
                         bin_path=None,
                         mpirun=mpirun,
                         qgrid='64 64 64',
                         thirdorder=False, 
                         stochastic=True,
                         quantum=True)
        fe = tdp.make_anharmonic_free_energy(**fe_params)
        
        # extract the number of configurations
        with open(conv_dir.joinpath('infile.meta').absolute(), 'r') as fl:
            lines = fl.readlines()
        nconfs = int(lines[1].split()[0])
        
        conv_prop.append([nconfs, fe])
        
        if len(conv_prop) < 2:
            return [False, 0]
        if conv_prop[-2][1] == 0:
            return [False, 0]
        else:
            err = (conv_prop[-1][1] - conv_prop[-2][1])/abs(conv_prop[-2][1])
        if abs(err) < thr:
            return [True, err]
        else:
            return [False, err]
        
        

    def plot_convergence(conv_prop):
        point_to_plot = np.array([[x[0], x[1]]for x in conv_prop])
        fig = plt.figure()
        plt.plot(point_to_plot[:,0], point_to_plot[:,1], '.')
        plt.xlabel('N. of configurations used')
        plt.ylabel('free energy ($eV$)')
        plt.savefig(fname='convergence_free_energy.png', bbox_inches='tight', dpi=600, format='png')
        plt.close()
        

    def save_ifc(path, save_name, dir):
        # save the outfile.forceconstant given by path into a separate folder (dir)
        os.system(f'cp {Path(path).absolute()} {Path(dir).joinpath(save_name).absolute()}')

    def nconfs_to_gen(n_iter, offset):
        '''
        Function that determines the number of configurations to generate at each iteration, based on the number
        of iteration already done (n_iter).
        '''
        #limits = [[10, 10], [20, 20], [30, 50], [50, 70], [70, 100], [100, 200], [500, 1000], [1000, 5000, 10000]
        #for limit in limits:
        #    if n_iter < limit:
        #    return limit
        #nconfs = offset
        #return offset + 2*n_iter
        return int(offset + n_iter**(0.18 * n_iter**0.5))

    def save_prop():
        save_free_energy()
     
    def save_free_energy():
        explain = 'This file contains two variables:\n1. This one, that you are currently reading;\n2. A list that contains, as elements, a list of two objects. The first object is the number of configurations and the second object is the free energy in eV'
        with open(Path('free_energies.pkl').absolute(), 'wb') as fl:
                pkl.dump([explain, conv_prop], fl)
    
    def save_msd():
        explain = 'This file contains two variables:\n1. This one, that you are currently reading;\n2. A list that contains, as elements, a list of two objects. The first object is the number of configurations and the second object is a list containing, in order, the msd along x, y, z and "absolute".'
        with open(Path('mean_squared_displacements.pkl').absolute(), 'wb') as fl:
                pkl.dump([explain, msds], fl)

    def save_and_exit():
        save_prop()
        exit()

    
    ##### END FUNCTIONS #####
    
    root_dir=Path(root_dir)
    if mlp_bin_path is not None:
        mlp_bin_path = Path(mlp_bin_path)
    if mlp_pot_path is not None:
        mlp_pot_path = Path(mlp_pot_path)
                
    # TO START WE NEED:
    # - the unit cell
    # - the supercell
    # OPTIONAL:
    # - the LO-TO splitting file (if necessary)
    # - an ifc infile
    
    # initialise the log file
    log_path = root_dir.joinpath('stdep_log.out')
    l = logger(log_path.absolute())
    
    run_MTP = compute_props_MTP

     
    conv_prop = []

    n_confs_done = 0

    if first_order == True:
        first_order = '--firstorder'
    else:
        first_order = ''
        
    if not root_dir.joinpath('infile.ucposcar').exists():
        l.log(f'File {root_dir.joinpath("infile.ucposcar")} does not exist!')
        exit()
    else:
        uc_path = root_dir.joinpath('infile.ucposcar')

    if not root_dir.joinpath('infile.ssposcar').exists():
        l.log(f'File {root_dir.joinpath("infile.ssposcar")} does not exist!')
        exit()
    else:
        ss_path = root_dir.joinpath('infile.ssposcar')


    if loto == True:
        if not root_dir.joinpath('infile.lotosplitting').exists():
            l.log(f'You asked to apply LO-TO splitting, but file {root_dir.joinpath("infile.lotosplitting")} does not exist!')
            exit()
        else:
            loto_path = root_dir.joinpath('infile.lotosplitting')
    if preexisting_ifc == True:
        if not root_dir.joinpath('infile.forceconstant').exists():
            l.log(f'You asked to use pre-existing ifcs, but file {root_dir.joinpath("infile.forceconstant")} does not exist!')
            exit()
        else:
            start_ifc_path = root_dir.joinpath('infile.forceconstant')
    else:
        if max_freq == None:
            l.log(f'You asked not to use any pre-existing ifcs, but you did not provide a maximum frequency!')
            exit()

    confs_dir = root_dir.joinpath('configurations')
    confs_dir.mkdir(parents=True, exist_ok=True)

    ln_s_f(uc_path, confs_dir)
    ln_s_f(ss_path, confs_dir)

    if preexisting_ifc == True:
        ln_s_f(start_ifc_path, confs_dir)


    # reference structure
    ref_at = read(ss_path.absolute(), format='vasp')

    # GENERATE FIRST CONFIGURATION
    new_confs_name = 'new_confs.traj'

    if preexisting_ifc == True:
        tdp.make_canonical_configurations(nconf=1, temp=T, quantum=quantum, dir=confs_dir.absolute(), outfile_name=new_confs_name, pref_bin='')
    else:
        tdp.make_canonical_configurations(nconf=1, temp=T, quantum=quantum, max_freq=max_freq, dir=confs_dir.absolute(), outfile_name=new_confs_name, pref_bin='')
    n_confs_done += 1
    l.log(f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
    l.log(f'Iteration n. 1 started.')
    l.log(f'The first configuration has been generated.')
    new_confs_path = confs_dir.joinpath(new_confs_name)
    
    # COMPUTE TRUE PROPERTIES
    ats = read(new_confs_path.absolute(), index=':')
    l.log(f'Compute true props for {len(ats)} confs.')
    confs_name = 'configurations.traj' # name of the file containing the old structures that will be appended with the new ones + props
    confs_path = confs_dir.joinpath(confs_name)
    mlip_dir = root_dir.joinpath('mlip')
    mlip_dir.mkdir(parents=True, exist_ok=True)

    params_to_compute = dict(mlp_bin=mlp_bin_path.absolute(), mlp_pot=mlp_pot_path.absolute(), mpirun='mpirun')

    compute_true_props(dir=mlip_dir.absolute(), atoms=ats, confs_path=confs_path.absolute(), function_to_compute=run_MTP, params_to_compute=params_to_compute)

        # this creates an ase.Atoms object with struct + props of old + new confs           
        # called {confs_name}
        # now configurations.traj contains (old + new) (structs + props)
    
    
    
    # EXTRACT IFCs; we don't need third-order ifcs, because canonical configurations doesn't consider them
    ifc_dir = root_dir.joinpath('ifc')
    ifc_dir.mkdir(parents=True, exist_ok=True)
    ifc_save_dir = root_dir.joinpath('ifc_savings')
    ifc_save_dir.mkdir(parents=True, exist_ok=True)

    # now we make the infiles
    ln_s_f(uc_path, ifc_dir)
    ln_s_f(ss_path, ifc_dir)
    ats = read(confs_path.absolute(), index=':')

    tdp.make_forces(ats, ifc_dir.absolute())
    tdp.make_positions(ats, ifc_dir.absolute())
    tdp.make_stat(ats, ifc_dir.absolute())
    tdp.make_meta(ats, ifc_dir.absolute(), temp=T)
    
    
    if loto == True:
        ln_s_f(loto_path, ifc_dir)
        polar = '--polar'
    else:
        polar = ''

    cmd = f'{mpirun} {tdp_bin_dir.joinpath("extract_forceconstants").absolute()} -rc2 {rc2} -rc3 {rc3} {polar} -U0 {first_order}'
    logpath = ifc_dir.joinpath("log_ifc")
    errpath = ifc_dir.joinpath("err_ifc")
    with open(logpath.absolute(), 'w') as log, open(errpath.absolute(), 'w') as err:
        run(cmd.split(), cwd=ifc_dir.absolute(), stdout=log, stderr=err)
    ifc_path = ifc_dir.joinpath('outfile.forceconstant')
    ifc_save_name = 'outfile.forceconstant_0'
    save_ifc(ifc_path.absolute(), ifc_save_name, ifc_save_dir.absolute())
    l.log(f'Interatomic force constants have been extracted from the set of ({len(ats)}) configurations.')
    l.log(f'----------------------------')
    
    for n_iter in range(max_iter):
        l.log(f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-')
        l.log(f'Iteration n. {n_iter + 2} started.')
        ln_s_f(ifc_dir.joinpath('outfile.forceconstant'), confs_dir.joinpath('infile.forceconstant'))
        nconfs = nconfs_to_gen(n_iter, nconfs_start) 
        tdp.make_canonical_configurations(nconf=nconfs, temp=T, quantum=quantum, dir=confs_dir.absolute(), outfile_name=new_confs_name, pref_bin='')
        
        n_confs_done += nconfs
        l.log(f'{nconfs} new configurations have been generated.')
        
        # compute en, for, str on the conf
        ats = read(new_confs_path.absolute(), index=':')
        #print(f'Just read {confs_path}: it has {len(ats)} confs')
        compute_true_props(dir=mlip_dir, atoms=ats, confs_path=confs_path.absolute(), function_to_compute= run_MTP, params_to_compute=params_to_compute)

        # EXTRACT IFCs; we don't need third-order ifcs, because canonical configurations doesn't consider them
        ats = read(confs_path.absolute(), index=':')
        
        tdp.make_forces(ats, ifc_dir.absolute())
        tdp.make_positions(ats, ifc_dir.absolute())
        tdp.make_stat(ats, ifc_dir.absolute())
        tdp.make_meta(ats, ifc_dir.absolute(), temp=T)
        
        logpath = Path('log_ifc')
        errpath = Path('err_ifc')
        with open(logpath.absolute(), 'w') as log, open(errpath.absolute(), 'w') as err:
            run(cmd.split(), cwd=ifc_dir.absolute(), stdout=log, stderr=err)
        ifc_save_name = f'outfile.forceconstant_{n_iter + 1}'
        save_ifc(ifc_path.absolute(), ifc_save_name, ifc_save_dir.absolute())
        
        l.log(f'Interatomic force constants have been extracted from the set of ({len(ats)}) configurations.')
        l.log(f'The check of the convergence criterion started.')
        # now let's check convergence
        conv = check_convergence(root_dir.absolute(), conv_prop, confs_path.absolute(), thr=thr)
        #l.log(str(conv[1]))
        conv = conv[0]
        #plot_convergence(conv_prop)
        save_prop()
        
        if conv == True:
            l.log(f'The configurations converged after the {n_confs_done}-th one. Free energy: {conv_prop[-1][1]} eV')
            exit()

        if n_confs_done >= max_confs:
            l.log(f'Maximum number of configurations reached ({max_confs}) but the configurations were not converged.')
            exit()
            
        l.log(f'---> still not converged.')
        l.log(f'{n_confs_done} configurations have been done in {n_iter} iterations.')
        l.log(f'----------------------------')

    if conv == False:
        l.log('The maximum number of iteration was reached but the calculation was not converged.')
        exit()

def launch_stdep(
    root_dir: str = './',
    ucell=None,
    make_supercell: bool = False,
    scell_mat=None,
    scell=None,
    T: float = None,
    preexisting_ifcs: bool = False,
    preexisting_ifcs_path=None,
    max_freq: float = None,
    quantum: bool = True,
    tdep_bin_directory: str = None,
    first_order: bool = True,
    displ_threshold_firstorder: float = 0.0001,
    max_iterations_first_order: int = None,
    rc2s: List[float] = None,
    rc3: float = None,
    polar: bool = False,
    loto_infile: str = None,
    ifc_max_err_threshold: float = None,
    nconfs: List[int] = None,
    pref_bin: str = '',
    mlip_pot_path: str = None,
    mlip_bin: str = None
):
    """
    Launches the sTDEP iterative workflow for force constant extraction and refinement.

    At each iteration, the second-order interatomic force constants (IFCs) are converged
    with respect to the cutoff (rc2). The converged IFCs from the current iteration are then
    used to generate the sampling structures for the next iteration.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root working directory. Default is './'.

    ucell : ase.atoms.Atoms
        The unit cell atoms object.

    make_supercell : bool, optional
        If True, the supercell will be generated using `scell_mat`.
        If False, the provided `scell` will be used directly.
        Default is False.

    scell_mat : array-like, optional
        Transformation matrix to build the supercell from the unit cell.

    scell : ase.atoms.Atoms, optional
        The supercell. Must be obtained by left-multiplying the unit cell by a matrix.

    T : float
        Temperature in Kelvin.

    preexisting_ifcs : bool, optional
        If True, pre-existing IFCs will be used in the first iteration (`preexisting_ifcs_path` is required).
        If False, the Debye model will be used (`max_freq` is required).
        Default is False.

    preexisting_ifcs_path : str or pathlib.Path, optional
        Path to the pre-existing IFCs.

    max_freq : float, optional
        Maximum phonon frequency (in THz) to be used in the Debye model if `preexisting_ifcs` is False.

    quantum : bool, optional
        If True, the Bose-Einstein distribution is used for phonon sampling.
        Default is True.

    tdep_bin_directory : str or pathlib.Path, optional
        Directory containing the TDEP binaries. If None, the default system installation will be used.

    first_order : bool, optional
        If True, the first-order TDEP correction will be applied iteratively to the unit cell positions.
        Default is True

    displ_threshold_firstorder : float, optional
        Convergence threshold (in Å) for atomic displacements during the TDEP first-order optimization.

    max_iterations_first_order : int, optional
        Maximum number of iterations allowed in the TDEP first-order optimization loop.

    rc2s : list of float
        List of second-order interaction cutoff values (in Å). The IFCs will be calculated for each cutoff to assess their convergence.
    
    rc3 : float, optional
        Third-order interaction cutoff (in Å). May be unused depending on setup.

    polar : bool, optional
        LO-TO splitting. If True, loto_infile must be given
    
    loto_infile : str or pathlib.Path, optional
        Filepath to the loto splitting infile
    
    ifc_max_err_threshold : float, optional
        Maximum allowed error threshold (in eV/Å²) to consider the IFCs converged at each iteration.
        Default if 0.0001.

    nconfs : list of int, optional
        List of the number of sampling configurations to be generated at each sTDEP iteration.
        This will determine also the number of iterations. Each entry defines the number of configurations
        for that iteration.

    pref_bin : str, optional
        String to prefix before the binary execution command (e.g., 'mpirun -n 32').
        Default is nothing.

    mlip_pot_path : str or pathlib.Path
        Path to the MTP `.pot` potential file.

    mlip_bin : str or pathlib.Path
        Path to the MTP binary executable.

    Notes
    -----
    - This function does not return any value but orchestrates the sTDEP workflow inside the given `root_dir`.
    - Make sure all provided paths are valid and the necessary binaries are accessible.
    - The combination of `make_supercell` and the provided inputs must be consistent, otherwise errors will occur.

    """
   
    # sTDEP steps:
    # 1. first iteration
    #     1.1 sample 4 configurations with the initial ifcs 
    #     1.2 compute properties
    #     1.3 extract ifcs: first order thing + convergence w.r.t rc2
    # 2. loop over the iterations
    #     2.i.1 sample N(i) configurations with the previous ifcs
    #     2.i.2 compute properties
    #     2.i.3 extract ifcs - convergence w.r.t rc2
    #     2.i.4 assess convergence
    # 3. final tidying of the files and sampling

    if make_supercell == False:
        if scell is None:
            raise TypeError('Since make_supercell is False, you must provide a supercell')
    else:
        if scell_mat is None:
            raise TypeError('Since make_supercell is True, you must provide scell_mat!')
        else:
            scell = mk_supercell(ucell, scell_mat) 
    if preexisting_ifcs is False:
        if max_freq is None:
            raise TypeError('Since preexisting_ifcs is False, you must provide max_freq!')
    else:
        preexisting_ifcs_path = Path(preexisting_ifcs_path)
        if not preexisting_ifcs_path.is_file():
            raise ValueError(f'The file {preexisting_ifcs_path.absolute()} does not exist!')
        max_freq = False
    if tdep_bin_directory is not None:
        tdep_bin_directory = Path(tdep_bin_directory)
    
    if polar == True:
        if loto_infile is None:
            raise TypeError('Since polar is True, you must provide loto_infile!')
        else:
            loto_infile = Path(loto_infile) 

    niters = len(nconfs)

    root_dir = Path(root_dir)

    # infiles
    infiles_dir = root_dir.joinpath('infiles')
    infiles_dir.mkdir(parents=True, exist_ok=True)
    write(infiles_dir.joinpath('infile.ucposcar'), ucell, format='vasp')
    write(infiles_dir.joinpath('infile.ssposcar'), scell, format='vasp')

    if preexisting_ifcs == True:
        shutil.copy(preexisting_ifcs_path, infiles_dir.joinpath('infile.forceconstant'))

    iters_dir = root_dir.joinpath('iterations')
    iters_dir.mkdir(parents=True, exist_ok=True)

    print('++++++++++++++++++++++++++++++++++++++')
    print('----------- sTDEP launched -----------')
    print('++++++++++++++++++++++++++++++++++++++')

    # 1. first iteration
    for iter in range(1, niters+1):
        print(f'====== ITERATION n. {iter} ======')
        iter_dir = iters_dir.joinpath(f'iter_{iter}')
        iter_dir.mkdir(parents=True, exist_ok=True)

        #   1.1 sample 4 configurations with the initial ifcs
        make_canonical_configurations_parameters = dict(ucell = ucell,
                                                        scell = scell,
                                                        nconf = nconfs[iter-1],
                                                        temp = T,
                                                        quantum = quantum,
                                                        dir = iter_dir,
                                                        outfile_name = 'new_confs.traj', # this will be saved inside dir
                                                        pref_bin=pref_bin,
                                                        tdep_bin_directory=tdep_bin_directory)
        
        if iter == 1:
            if preexisting_ifcs == False:
                make_canonical_configurations_parameters['max_freq'] = max_freq
            else:
                make_canonical_configurations_parameters['ifcfile_path'] = preexisting_ifcs_path
        else:
            make_canonical_configurations_parameters['ifcfile_path'] = last_ifc_path 
        
        tdp.make_canonical_configurations(**make_canonical_configurations_parameters)

        latest_confs = read(iter_dir.joinpath('new_confs.traj'), index=':')

        #   1.2 1.2 compute properties
        prop_iter = iter_dir.joinpath('true_props')
        prop_iter.mkdir(exist_ok=True, parents=True)

        latest_confs_computed = mlp.calc_efs_from_ase(mlip_bin = mlip_bin, 
                                                    atoms = latest_confs, 
                                                    mpirun = 'mpirun -n 6', 
                                                    pot_path = mlip_pot_path, 
                                                    cfg_files=False, 
                                                    dir = prop_iter,
                                                    write_conf = True, 
                                                    outconf_name = 'new_confs_computed.traj')
        latest_confs_computed_path = prop_iter.joinpath('new_confs_computed.traj')

        ifc_dir = iter_dir.joinpath('ifc')
        ifc_dir.mkdir(parents=True, exist_ok=True)
        
        min_dist = min([min_distance_to_surface(x.get_cell()) for x in latest_confs_computed]) ##


        
        last_ifc_path, max_diffs, avg_diffs = tdp.conv_rc2_extract_ifcs(unitcell = ucell,
                                                                        supercell = scell,
                                                                        sampling = latest_confs_computed,
                                                                        timestep = 1,
                                                                        dir = ifc_dir,
                                                                        first_order = first_order,
                                                                        displ_threshold_firstorder = displ_threshold_firstorder,
                                                                        max_iterations_first_order = max_iterations_first_order,
                                                                        rc2s = rc2s, 
                                                                        rc3 = rc3, 
                                                                        polar = polar,
                                                                        loto_filepath = loto_infile,
                                                                        stride = 1, 
                                                                        temperature = T,
                                                                        bin_prefix = pref_bin,
                                                                        tdep_bin_directory = tdep_bin_directory,
                                                                        max_err_threshold = ifc_max_err_threshold)
        
        shutil.copy(last_ifc_path, iter_dir.joinpath('converged_outfile.forceconstant'))
        print(f'============================')

    # check if IFC convergence through the iterations
    ifcs = []
    n_structs = []
    for iter in range(1, niters+1):
        iter_dir = iters_dir.joinpath(f'iter_{iter}')
        unitcell = read(iter_dir.joinpath('infile.ucposcar'), format='vasp')
        supercell = read(iter_dir.joinpath('infile.ssposcar'), format='vasp')
        ifc = tdp.parse_outfile_forceconstants(iter_dir.joinpath('converged_outfile.forceconstant'), unitcell, supercell)
        ifcs.append(ifc)
    ifcs = np.array(ifcs)
    ifcs = np.array(ifcs)
    diffs = np.abs(ifcs[1:] - ifcs[:-1])
    max_diffs = np.max(diffs, axis=(1, 2, 3, 4))
    Fig = plt.figure(figsize=(15,4))
    plt.plot(nconfs[1:], max_diffs, '.')
    plt.title('IFC convergence: max abs. error')
    plt.ylabel('Error on the IFCs (eV/$\mathrm{\AA}^2$)')
    plt.xlabel('Number of structures')
    figpath = root_dir.joinpath(f'Convergence.png')
    plt.savefig(fname=figpath, bbox_inches='tight', dpi=600, format='png')

    
def conv_iterations(root_dir, nconfs, iters_dir, verbose=True):
    def fake_print(a):
        pass
    if verbose == False:
        print = fake_print 
    ifcs = []
    n_structs = []
    nconfs_actual = []
    niters = len(nconfs)
    for iter in range(1, niters+1):
        iter_dir = iters_dir.joinpath(f'iter_{iter}')
        ifc_filepath = iter_dir.joinpath('converged_outfile.forceconstant')
        if not ifc_filepath.is_file():
            print(f'Iteration {iter} is not complete (converged ifcs missing). It will be skipped.')
            continue 
        unitcell = read(iter_dir.joinpath('infile.ucposcar'), format='vasp')
        supercell = read(iter_dir.joinpath('infile.ssposcar'), format='vasp')
        ifc = tdp.parse_outfile_forceconstants(ifc_filepath, unitcell, supercell)
        ifcs.append(ifc)
        nconfs_actual.append(nconfs[iter-1])

    if len(ifcs) < 2:
        print('Less than 2 iterations were complete! Aborting.')
        return
    
    ifcs = np.array(ifcs)
    ifcs = np.array(ifcs)
    diffs = np.abs(ifcs[1:] - ifcs[:-1])
    max_diffs = np.max(diffs, axis=(1, 2, 3, 4))
    avg_diffs = np.mean(diffs, axis=(1, 2, 3, 4))
    Fig = plt.figure(figsize=(15,11))
    Fig.add_subplot(2,1,1)
    plt.plot(nconfs_actual[1:], max_diffs, '.')
    plt.title('IFC convergence: max abs. error')
    plt.ylabel('Error on the IFCs (eV/$\mathrm{\AA}^2$)')
    plt.xlabel('Number of structures')

    Fig.add_subplot(2,1,2)
    plt.plot(nconfs_actual[1:], avg_diffs, '.')
    plt.title('IFC convergence: avg abs. error')
    plt.ylabel('Error on the IFCs (eV/$\mathrm{\AA}^2$)')
    plt.xlabel('Number of structures')

    figpath = root_dir.joinpath(f'Convergence.png')
    plt.savefig(fname=figpath, bbox_inches='tight', dpi=600, format='png')
    plt.close()



def conv_sizes(size_folders=None, labels=None, tdep_bin_directory=None, bin_pref='', outimage_path='Convergence_wrt_size.png'):
    
    freqs = []
    doss = []
    size_folders = [Path(folder) for folder in size_folders]
    iters_dirs = [folder.joinpath('iterations') for folder in size_folders]
   
    # Filter out non-existing directories
    valid_sizes = []
    for size_folder, iters_dir, label in zip(size_folders, iters_dirs, labels):
        if size_folder.is_dir():
            valid_sizes.append((size_folder, iters_dir, label))
        else:
            print(f'{size_folder.name}/iterations does not exist! It will be ignored.')

    # Re-extract valid folders and iteration directories
    size_folders = [fp[0] for fp in valid_sizes]
    iters_dirs = [fp[1] for fp in valid_sizes]
    labels = [fp[2] for fp in valid_sizes]


    iters = [[int(str(x).split('_')[-1])
                for x in iters_dir.glob('iter_*') 
                if x.joinpath('converged_outfile.forceconstant').is_file()]
            for iters_dir in iters_dirs]
    # Find folders with no converged iterations    
    to_pop = []
    for i, itlist in enumerate(iters):
        if len(itlist) == 0:
            print(f'{iters_dirs[i].parent.name} has no converged iteration, it will be ignored.')
            to_pop.append(i)
    
    # Remove them from all lists
    for i in reversed(to_pop):
        iters.pop(i)
        iters_dirs.pop(i)
        size_folders.pop(i)
        labels.pop(i)


    n_iters = [max(x) for x in iters] 
    
    
    max_iter = min(n_iters) # the maximum common number of iterations (the minimum among n_iters)
    
    for i, iter_dir in enumerate(iters_dirs):
        max_iter_dir = iter_dir.joinpath(f'iter_{max_iter}')
        ucell = read(max_iter_dir.joinpath('infile.ucposcar'), format='vasp')
        scell = read(max_iter_dir.joinpath('infile.ssposcar'), format='vasp')
        ifc_file = max_iter_dir.joinpath('converged_outfile.forceconstant').absolute()
        
        ph_dir = max_iter_dir.joinpath('phonons')
        ph_dir.mkdir(parents=True, exist_ok=True)

        tdp.run_phonons(dir=ph_dir, ucell=ucell, scell=scell, ifc_file=ifc_file, qgrid=32, dos=True, tdep_bin_directory=None, bin_pref=bin_pref, units='thz')
        print(ph_dir.joinpath('outfile.phonon_dos.hdf5'))
        fl = h5py.File(ph_dir.joinpath('outfile.phonon_dos.hdf5'))
        freqs.append(np.array(fl['frequencies']))
        doss.append(np.array(fl['dos']))

    for i, label in enumerate(labels):
        plt.plot(freqs[i], doss[i], label=label, lw=0.5)
        plt.xlabel(f'Frequency (THz)')
        plt.ylabel(f'Phonon DOS (states/THz)')
        plt.legend()
        plt.savefig(fname=outimage_path, bbox_inches='tight', dpi=600)
    plt.close()
    

def conv_iters_and_sizes(size_dirs, size_labels, outimage_path='./Convergence_wrt_size.png'):
    outimage_path = Path(outimage_path)
    size_dirs = [Path(x) for x in size_dirs]
    for i_s, size in enumerate(size_labels):
        size_dir = size_dirs[i_s]
        if not size_dir.is_dir():
            print(f'\t{size_dir.name} - not found')
            continue
        print(f'\t{size_dir.name} - found')
        iters_dir = size_dir.joinpath('iterations')
        iter_dirs = sorted([x for x in iters_dir.glob('iter_*')], key=lambda x: int(x.name.split('_')[-1]))
        iter_dirs_complete = [x.joinpath('converged_outfile.forceconstant').is_file() for x in iter_dirs]
        nconfs = []
        for i_i, iter_dir in enumerate(iter_dirs):
            if iter_dir.joinpath('new_confs.traj').is_file():
                nconfs.append(len(read(iter_dir.joinpath('new_confs.traj'), index=':')))
            else:
                nconfs.append(0)
            if iter_dirs_complete[i_i]:
                print(f'\t\t{iter_dir.name} - complete')
            else:
                print(f'\t\t{iter_dir.name} - incomplete (ignored)')
        conv_iterations(root_dir=size_dir, nconfs=nconfs, iters_dir=iters_dir, verbose=False)
    conv_sizes(size_folders=size_dirs, labels=size_labels, tdep_bin_directory=None, bin_pref='mpirun -n 6', outimage_path=outimage_path)