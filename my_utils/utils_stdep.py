import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from ase.atoms import Atoms
from ase.io import read, write
from ase.build import make_supercell
import os
import sys
sys.path.append('/scratch/ulg/matnan/slongo/my_scripts/')
import utils_tdep as tdp
from subprocess import run
import utils_mlip as mlp
from ase.calculators.singlepoint import SinglePointCalculator


class logger():
    def __init__(self, filepath):
        self.filepath = filepath
        if os.path.exists(filepath):
            os.system(f'rm {filepath}')
    
    def log(self, text):
        text = f'{text}\n'
        print(text)
        with open(self.filepath, 'a') as fl:
            fl.write(text)
            
    
    
def compute_props_MTP(dir, atoms=None, mlp_bin=None, mlp_pot=None):
    '''
    Function to compute properties with MTP. It also save the struct + props as an ase traj called new_confs.traj
    Args:
    dir(str): directory where MTP will run
    atoms(ase trajectory): list of ase.Atoms objects to compute properties of
    mlp_bin(str): path to the binary of MTP
    mlp_pot(str): path to the trained potential (.mtp) file to use
    '''
    # first we have to set the input files for MTP

    # 1. We need to convert the ase confs to a .cfg file
    #print(f'Just starting the MTP. {len(atoms)} confs have been provided.')
    file_path = f'{dir}in.cfg'
    mlp.conv_ase_to_mlip2(atoms=atoms, out_path=file_path, props=False)
    file_path2 = f'{dir}out.cfg'
    mlpbin = '/scratch/users/s/l/slongo/codes/mlip-2/build1/mlp'
    mlp.calc_efs(mlip_bin=mlp_bin, confs_path=file_path, pot_path=mlp_pot, out_path=file_path2, dir=dir)
    # now we have to store the struct + props as an ase trajectory
    # first, we know that "atoms" contains the ase traj without properties, so we only need to add the props.
    # Hence, we extract the properties from the out.cfg

    energy, forces, stress = mlp.extract_prop(file_path2)
    #print(f'Properties have been extracted. I got {len(energy)} energies.')
    if not len(energy) == len(atoms):
        print(f'Hey! not same lenght!: len energy is {len(energy)} and len atoms is {len(atoms)}')
        exit()
    for i, conf in enumerate(atoms):
        natoms = len(conf)
        calc = SinglePointCalculator(conf, energy=energy[i]*natoms, forces=forces[i], stress=stress[i])
        conf.calc = calc
    write(f'{dir}new_confs.traj', atoms)

        
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

    # compute props and save the new structs
    function_to_compute(dir, atoms, **params_to_compute)
    # now new_confs.traj contains new_struct + props, we can merge with the existing ones
    new_ats = read(f'{dir}new_confs.traj', index=':')
    ats_to_save = []
    if os.path.exists(confs_path):
        existing_confs = read(confs_path, index=':')
        ats_to_save.extend(existing_confs)
    ats_to_save.extend(new_ats)
    write(confs_path, ats_to_save)

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

def run_stdep(root_dir=os.path.abspath('./'),
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
              mlp_pot_path = None):
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
        return check_convergence_free_energy(root_dir=root_dir, conv_prop=conv_prop, thr=thr)
    
    
    def check_convergence_free_energy(root_dir, conv_prop, thr):
        '''
        The anharmonic free energy will be computed up to the second order and the convergence checked with respect to the
        previous one.
        Args:
        root_dir(str, path): root directory where the directory 'ifc' is.
        '''
        
        conv_dir = f'{root_dir}conv_free_energy/'
        os.system(f'mkdir -p {conv_dir}')
        
        # first we need to copy the infiles
        os.system(f'ln -s -f {root_dir}ifc/infile.stat {conv_dir}infile.stat')
        os.system(f'ln -s -f {root_dir}ifc/infile.meta {conv_dir}infile.meta')
        os.system(f'ln -s -f {root_dir}ifc/infile.positions {conv_dir}infile.positions')
        os.system(f'ln -s -f {root_dir}ifc/infile.forces {conv_dir}infile.forces')
        os.system(f'ln -s -f {root_dir}ifc/outfile.forceconstant {conv_dir}infile.forceconstant')
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
        with open(f'{conv_dir}infile.meta', 'r') as fl:
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
        
        

    def plot_convergence(prop_conv):
        point_to_plot = np.array([[x[0], x[1]]for x in prop_conv])
        fig = plt.figure()
        plt.plot(point_to_plot[:,0], point_to_plot[:,1], '.')
        plt.xlabel('N. of configurations used')
        plt.ylabel('free energy ($eV$)')
        plt.savefig(fname='convergence_free_energy.png', bbox_inches='tight', dpi=600, format='png')
        plt.close()
        

    def save_ifc(path, save_name, dir):
        # save the outfile.forceconstant given by path into a separate folder (dir)
        os.system(f'cp {path} {dir}{save_name}')

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
        return offset + 2*n_iter
    
    def save_prop():
        save_free_energy()
     
    def save_free_energy():
        explain = 'This file contains two variables:\n1. This one, that you are currently reading;\n2. A list that contains, as elements, a list of two objects. The first object is the number of configurations and the second object is the free energy in eV'
        with open('free_energies.pkl', 'wb') as fl:
                pkl.dump([explain, prop_conv], fl)
    
    def save_msd():
        explain = 'This file contains two variables:\n1. This one, that you are currently reading;\n2. A list that contains, as elements, a list of two objects. The first object is the number of configurations and the second object is a list containing, in order, the msd along x, y, z and "absolute".'
        with open('mean_squared_displacements.pkl', 'wb') as fl:
                pkl.dump([explain, msds], fl)

    def save_and_exit():
        save_prop()
        exit()

    
    ##### END FUNCTIONS #####
    
    
    # TO START WE NEED:
    # - the unit cell
    # - the supercell
    # OPTIONAL:
    # - the LO-TO splitting file (if necessary)
    # - an ifc infile
    
    # initialise the log file
    log_path = f'{root_dir}stdep_log.out'
    l = logger(log_path)
    
    run_MTP = compute_props_MTP

     
    conv_prop = []

    n_confs_done = 0

    if first_order == True:
        first_order = '--firstorder'
    else:
        first_order = ''
        
    if not os.path.exists(f'{root_dir}infile.ucposcar'):
        l.log(f'File {root_dir}infile.ucposcar does not exist!')
        exit()
    else:
        uc_path = f'{root_dir}infile.ucposcar'

    if not os.path.exists(f'{root_dir}infile.ssposcar'):
        l.log(f'File {root_dir}infile.ssposcar does not exist!')
        exit()
    else:
        ss_path = f'{root_dir}infile.ssposcar'


    if loto == True:
        if not os.path.exists(f'{root_dir}infile.lotosplitting'):
            l.log(f'You asked to apply LO-TO splitting, but file {root_dir}infile.lotosplitting does not exist!')
            exit()
        else:
            loto_path = f'{root_dir}infile.lotosplitting'
    if preexisting_ifc == True:
        if not os.path.exists(f'{root_dir}infile.forceconstant'):
            l.log(f'You asked to use pre-existing ifcs, but file {root_dir}infile.forceconstant does not exist!')
            exit()
        else:
            start_ifc_path = f'{root_dir}infile.forceconstant'
    else:
        if max_freq == None:
            l.log(f'You asked not to use any pre-existing ifcs, but you did not provide a maximum frequency!')
            exit()

    confs_dir = f'{root_dir}configurations/'
    os.system(f'mkdir -p {confs_dir}')

    os.system(f'ln -s -f {uc_path} {confs_dir}')
    os.system(f'ln -s -f {ss_path} {confs_dir}')

    if preexisting_ifc == True:
        os.system(f'ln -s -f {start_ifc_path} {confs_dir}')




    # reference structure
    ref_at = read(ss_path, format='vasp')

    # GENERATE FIRST CONFIGURATION
    new_confs_name = 'new_confs.traj'
    if preexisting_ifc == True:
        tdp.make_canonical_configurations(nconf=1, temp=T, quantum=quantum, dir=confs_dir, outfile_name=new_confs_name, pref_bin=mpirun)
    else:
        tdp.make_canonical_configurations(nconf=1, temp=T, quantum=quantum, max_freq=max_freq, dir=confs_dir, outfile_name=new_confs_name, pref_bin=mpirun)
    n_confs_done += 1
    new_confs_path = f'{confs_dir}{new_confs_name}' 


    # COMPUTE TRUE PROPERTIES
    ats = read(new_confs_path, index=':')
    l.log(f'Compute true props for {len(ats)} confs')
    confs_name = 'configurations.traj' # name of the file containing the old structures that will be appended with the new ones + props
    confs_path = f'{confs_dir}{confs_name}'
    mlip_dir = f'{root_dir}mlip/'
    os.system(f'mkdir -p {mlip_dir}')

    params_to_compute = dict(mlp_bin=mlp_bin_path, mlp_pot=mlp_pot_path)

    compute_true_props(dir=mlip_dir, atoms=ats, confs_path=confs_path, function_to_compute=run_MTP, params_to_compute=params_to_compute) # this creates an ase.Atoms object with struct + props of old + new confs                              # called {confs_name}
        # now configurations.traj contains (old + new) (structs + props)

    # EXTRACT IFCs; we don't need third-order ifcs, because canonical configurations doesn't consider them

    ifc_dir = f'{root_dir}ifc/'
    os.system(f'mkdir -p {ifc_dir}')
    ifc_save_dir = f'{root_dir}ifc_savings/'
    os.system(f'mkdir -p {ifc_save_dir}')

    # now we make the infiles
    os.system(f'ln -s -f {uc_path} {ifc_dir}')
    os.system(f'ln -s -f {ss_path} {ifc_dir}')
    ats = read(confs_path, index=':')

    tdp.make_forces(ats, ifc_dir)
    tdp.make_positions(ats, ifc_dir)
    tdp.make_stat(ats, ifc_dir)
    l.log(f'len ats: {len(ats)}')
    tdp.make_meta(ats, ifc_dir, temp=T)

    if loto == True:
        os.system(f'ln -s -f {loto_path} {ifc_dir}')
        polar = '--polar'
    else:
        polar = ''

    cmd = f'{mpirun} extract_forceconstants -rc2 {rc2} -rc3 {rc3} {polar} -U0 {first_order}'
    with open(f'{ifc_dir}ifc_log', 'w') as log, open(f'{ifc_dir}ifc_err', 'w') as err:
        run(cmd.split(), cwd=ifc_dir, stdout=log, stderr=err)
    ifc_path = f'{ifc_dir}outfile.forceconstant'
    ifc_save_name = 'outfile.forceconstant_0'
    save_ifc(ifc_path, ifc_save_name, ifc_save_dir)

    for n_iter in range(max_iter):
        l.log(f'Iteration n. {n_iter}')
        os.system(f'ln -s -f {ifc_path} {confs_dir}infile.forceconstant')
        nconfs = nconfs_to_gen(n_iter, nconfs_start) 
        tdp.make_canonical_configurations(nconf=nconfs, temp=T, quantum=quantum, dir=confs_dir, outfile_name=new_confs_name, pref_bin=mpirun)
        n_confs_done += nconfs
        l.log(f'made {nconfs} configs')
        # compute en, for, str on the conf
        ats = read(new_confs_path, index=':')
        #print(f'Just read {confs_path}: it has {len(ats)} confs')
        compute_true_props(dir=mlip_dir, atoms=ats, confs_path=confs_path, function_to_compute= run_MTP, params_to_compute=params_to_compute)

        # EXTRACT IFCs; we don't need third-order ifcs, because canonical configurations doesn't consider them
        ats = read(confs_path, index=':')

        tdp.make_forces(ats, ifc_dir)
        tdp.make_positions(ats, ifc_dir)
        tdp.make_stat(ats, ifc_dir)
        tdp.make_meta(ats, ifc_dir, temp=T)

        with open('ifc_log', 'w') as log, open('ifc_err', 'w') as err:
            run(cmd.split(), cwd=ifc_dir, stdout=log, stderr=err)
        ifc_save_name = f'outfile.forceconstant_{n_iter + 1}'
        save_ifc(ifc_path, ifc_save_name, ifc_save_dir)
        
        # now let's check convergence
        conv = check_convergence(conv_prop, confs_path=confs_path, root_dir=root_dir, thr=thr)
        
        l.log(str(conv[1]))
        conv = conv[0]
        plot_convergence(prop_conv)
        save_prop()
        
        if conv == True:
            l.log(f'The configurations converged after the {n_confs_done}-th one. Free energy: {prop_conv[-1][1]} eV')
            exit()

        if n_confs_done >= max_confs:
            l.log(f'Maximum number of configurations reached ({max_confs}) but the configurations were not converged.')
            exit()

        l.log(f'{n_confs_done} have been done in {n_iter} iterations.')

    if conv == False:
        l.log('The maximum number of iteration was reached but the calculation was not converged.')
        exit()

