import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import os

import sys

from .utils import data_reader, cap_first, repeat, warn, from_list_to_text, mae, rmse, R2, low_first, path
from .Graphics_matplotlib import Histogram


from ase.io import read, write
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from numbers import Number

from subprocess import run



def plot_correlations(dtset_ind, ind, dir='', offsets=None, save=False):
    '''
    This function plot the correlation graphs for energy, forces and stress,
    Parameters:
    dtset_ind (int): 0 for test set, 1 for training set
    ind (int): 0 for energy, 1 for forces, 2 for stress
    dir (str): path of the directory where the comparison files are
    '''
    dtset_used = ['Test', 'MLIP'] # MLIP corresponds to using the training set
    #dtset_ind = 0
    # order: 0=energy, 1=forces, 2=stress
    names = ['energy', 'forces', 'stress']
    units = ['eV/at', 'ev/$\mathrm{\AA}$', 'GPa']
    file_names = [f'{dir}{dtset_used[dtset_ind]}-{cap_first(x)}_comparison.dat' for x in names]
    if offsets is None:
        #offsets = [35, 1.8, 1] # negative offset of the y position of the text box 
        offsets = [0, 0, 0]
    #ind = 1

    data = data_reader(file_names[ind], separator="", skip_lines=0, frmt="str")

    rmse = data[0][2] 
    mae = data[0][5]
    #print(f"rmse {rmse}, mae {mae}")
    #print(file_names[ind])
    #print(data[0])
    
    
    ss_res = sum([(float(x[0])-float(x[1]))**2 for x in data[2:]])
    avg_DFT = sum([float(x[0]) for x in data[2:]])
    avg_DFT = avg_DFT/(len(data)-2)
    ss_tot = sum([(float(x[0]) - avg_DFT)**2 for x in data[2:]])
    R2 = 1 - ss_res/ss_tot
    txt = f"rmse: {rmse} {units[ind]}\nmae: {mae} {units[ind]}\nR$^2$: {str(round(R2, 4))}"
    x_data = np.array([float(pip[0]) for pip in data[2:]])
    y_data = np.array([float(pip[1]) for pip in data[2:]])
    n = 2.3
    fig1 = plt.figure(figsize=(n, n))
    ax = fig1.add_axes((0, 0, 1, 1))
    ax.plot(x_data, y_data, ".", markersize=10, mew=0.6, mec="#00316e", mfc='#70c8ff')
    ax.plot(x_data, x_data)
    ax.set_xlabel(f"DFT {names[ind]} ({units[ind]})")
    ax.set_ylabel(f"ML {names[ind]} ({units[ind]})")
    tbox = ax.text(min(x_data), max(y_data)-offsets[ind], txt, fontweight='bold')
    bbox = tbox.get_window_extent() # get the bounding box of the text
    transform = ax.transData.inverted() # prepare the transformation into data units
    bbox = transform.transform_bbox(bbox) # transform it into data units
    fname = f"{dtset_used[dtset_ind]}-{cap_first(names[ind])}"
    ax.set_title(fname)
    if save == True:
        os.system('mkdir -p Figures')
        plt.savefig('Figures/' + fname + ".png", format='png', dpi=600, bbox_inches='tight')
        plt.savefig('Figures/' + fname + '.svg', format='svg')

        

def conv_ase_to_mlip2(atoms, out_path, props=True):
    '''
    Convert a trajectory file of ASE into a .cfg file for mlip-2
    Arguments:
    atoms(list): list of the Atoms objects (configurations)
    out_path(str): path to the output file (must include the extension .cfg) 
    props(bool): if True energy, stress and forces will be copied too (they must have been computed for each configuration in
                 in the trajectory!); if False no property will be copied.
    '''
    if isinstance(atoms, Atoms):
        atoms = [atoms]
        
    text = ''
    for x in atoms:
        conf = x
        natom = len(conf)
        cell = conf.get_cell()
        if props == True:
            energy = conf.get_potential_energy()
            forces = conf.get_forces()
            stress = conf.get_stress() * conf.get_volume() # MTP uses stress multiplied by the volume
        at_syms = conf.get_chemical_symbols()
        positions = conf.get_positions()
        elemlist = list(set(at_syms))
        for i in range(len(elemlist)):
            elemlist[i] = cap_first(elemlist[i])
        elemlist.sort()
        nelem = len(elemlist)
        elems = dict()
        for i, el in enumerate(elemlist):
            elems[f'{el}'] = i

        # WRITE TEXT
        text += f'BEGIN_CFG\n'
        text += f'Size\n'
        text += f'{natom}\n'
        text += f'Supercell\n'
        text += f'{cell[0][0]}\t{cell[0][1]}\t{cell[0][2]}\n'
        text += f'{cell[1][0]}\t{cell[1][1]}\t{cell[1][2]}\n'
        text += f'{cell[2][0]}\t{cell[2][1]}\t{cell[2][2]}\n'
        text += f'AtomData:\tid\ttype\tcartes_x\tcartes_y\tcartes_z\tfx\tfy\tfz\n'
        for i, atm in enumerate(at_syms):
            text += f'{i+1}\t'
            text += str(elems[atm]) + '\t'
            #text += f'{elems[f'{atm}']}\t'
            text += f'{positions[i][0]:15.20f}\t'
            text += f'{positions[i][1]:15.20f}\t'
            text += f'{positions[i][2]:15.20f}\t'
            if props == True:
                text += f'{forces[i][0]:15.20f}\t'
                text += f'{forces[i][1]:15.20f}\t'
                text += f'{forces[i][2]:15.20f}\t'
            else:
                text += f'{0:15.20f}\t' # fake forces
                text += f'{0:15.20f}\t'
                text += f'{0:15.20f}\t'
            text += f'\n'
        if props == True:
            text += f'Energy\n'
            text += f'{energy:15.20f}\n'
            text += f'PlusStress:\txx\tyy\tzz\tyz\txz\txy\n'
            text += f'{-stress[0]:15.20f}\t{-stress[1]:15.20f}\t{-stress[2]:15.20f}\t{-stress[3]:15.20f}\t{-stress[4]:15.20f}'\
                  + f'\t{-stress[5]:15.20f}\n'
        text += f'END_CFG\n\n'
    with open(out_path, 'w') as fl:
        fl.write(text)
    print(f'File printed to {out_path}')



def check_sets(bins, trainset_name='TrainSet.traj', testset_name='TestSet.traj', save=False):
    '''
    Function to check the distribution of the temperatures among between the train and test set.
    Arguments:
    bins(numpy array): array with the temperatures to consider
    trainset_name(str): path to the train set
    testset_name(str): path to the test set
    save(bool): save the figures
    '''
    train = read(trainset_name, index=":")
    test = read(testset_name, index=":")

    t_train = [x.get_temperature() for x in train]
    t_test = [x.get_temperature() for x in test]
    binses = binses

    hist_train = Histogram(t_train)
    hist_train.histofy(mode='n_bins', nbins=len(binses), bins=binses, normalized=True)
    hist_train.make_bars(plot=False)
    hist_train.plot_bars(save=save, fname='Train_sets.png', format='png', dpi=600, bbox_inches='tight')

    hist_test = Histogram(t_test)
    hist_test.histofy(mode='custom_centers', bins=binses, normalized=True)
    hist_test.make_bars(plot=False)
    hist_test.plot_bars(save=save, fname='Test_sets.png', format='png', dpi=600, bbox_inches='tight')

    
    
    
    
def make_mtp_file(sp_count, mind, maxd, rad_bas_sz, rad_bas_type='RBChebyshev', lev=8, mtps_dir=None, wdir='./', out_name='init.mtp'):
    '''
    Function to create the .mtp file necessary for the training
    Args:
    sp_count(str): species_count; how many elements
    mind(float): min_dist; minimum distance between atoms
    maxd(float): max_dist; maximum distance between atoms
    rad_bas_sz(int): radial_basis_size; size of the radial basis
    rad_bas_type(str): radial_basis_type; set of radial basis (default=RBChebyshev)
    lev(int): MTP level; one of [2, 4,  6, 8, 10, 12, 14,1 6, 18, 20, 22, 24, 26, 28]
    mtps_dir(str): directory where the untrained .mtp files are stored for all levels
    wdir(str): path to the working directory (where the new .mtp file will be saved)
    out_name(str): name of the .mtp file (must include extension) (only name, no path)
    '''
    
    if mtps_dir == None:
        mtps_dir = '/scratch/ulg/matnan/slongo/codes/mlip-2/untrained_mtps'
    
    mtps_dir = os.path.abspath(mtps_dir)
    if not mtps_dir.endswith('/'):
        mtps_dir += '/'
    
    wdir = os.path.abspath(wdir)
    if not wdir.endswith('/'):
        wdir += '/'
        
    lev = int(lev)
    src_name = f'{lev:0>2d}.mtp'
    src_path = f'{mtps_dir}{src_name}'
    
    with open(src_path, 'r') as fl:
        lines = fl.readlines()
        
    for i, line in enumerate(lines):
        if 'species_count' in line:
            lines[i] = f'species_count = {sp_count}\n'
        elif 'radial_basis_type' in line:
            lines[i] = f'radial_basis_type = {rad_bas_type}\n'
        elif 'min_dist' in line:
            lines[i] = f'\tmin_dist = {mind}\n'
        elif 'max_dist' in line:
            lines[i] = f'\tmax_dist = {maxd}\n'
        elif 'radial_basis_size' in line:
            lines[i] = f'\tradial_basis_size = {rad_bas_sz}\n'
    outfile_path = f'{wdir}{out_name}'
    text = from_list_to_text(lines)
    with open(outfile_path, 'w') as fl:
        fl.write(text)

        
        
def extract_prop(filepath):
    '''
    Function to extract energy (per atom), forces and stress from a set of configurations contained in
    a single .cfg file.
    Arguments:
    filepath(str): path to the .cfg file containing the configurations
    Returns:
    energy(list): energy per atom of each configuration (in eV)
    forces(list): forces with shape confs x natoms x 3 (in eV/Angst)
    stress(list): stress tensor for each configuration (in eV/Angst^2); convention: sigma = + dE/dn (n=strain).
    '''
    with open(filepath, 'r') as fl:
        lines = fl.readlines()
        nlines = len(lines)
        nconf = 0
        iterator = iter(enumerate(lines))
        forces = []
        energy = []
        stress = []
        
        for i, line in iterator:
            if "BEGIN CFG" in line:
                nconf += 1
            if "Size" in line:
                natom = int(lines[i+1])
            if 'Supercell' in line:
                cell = []
                cell.append( [float(x) for x in lines[i+1].split()] )
                cell.append( [float(x) for x in lines[i+2].split()] )
                cell.append( [float(x) for x in lines[i+3].split()] )
                next(iterator)
                next(iterator)
                next(iterator)
                
            if "AtomData" in line:
                for j in range(natom):
                    forces.append(float(lines[i + 1 + j].split()[5]))
                    forces.append(float(lines[i + 1 + j].split()[6]))
                    forces.append(float(lines[i + 1 + j].split()[7]))
                for k in range(natom):
                    next(iterator)
                    
            if "Energy" in line:
                energy.append(float(lines[i+1])/natom) # ENERGY PER ATOM!!!!
            
            if "PlusStress" in line:
                at = Atoms(numbers=[1], cell=cell)
                V = at.get_volume()
                stress.append(-float(lines[i+1].split()[0])/V) # xx
                stress.append(-float(lines[i+1].split()[1])/V) # yy
                stress.append(-float(lines[i+1].split()[2])/V) # zz
                stress.append(-float(lines[i+1].split()[3])/V) # yz
                stress.append(-float(lines[i+1].split()[4])/V) # xz
                stress.append(-float(lines[i+1].split()[5])/V) # xy
        nconfs = len(energy)
        natoms = int(len(forces)/nconfs/3)
        forces = np.array(forces).reshape(nconfs, natoms, 3)
        stress = np.array(stress).reshape(nconfs, 6)
        energy = np.array(energy)
        return energy, forces, stress
    

    
    
    
    
def make_comparison(file1, file2, props='all', make_file=False, dir='', outfile_pref='', units=None):
    '''
    Create the comparison files for energy, forces and stress starting from the .cfg files.
    Arguments:
    file1(str): PATH to the file with the true values (.cfg)
    file2(str): PATH to the file with the ML values (.cfg)
    props(str, list): must be one value or a list of values chosen from ['energy', 'forces', 'stress', 'all'].
                     if a list is given containing 'all', all three properties will be considered, independent on
                     the other elements of the list
    make_file(bool): True: create a comparison file
    dir(str): directory the output file will be saved (if make_file=True)
    outfile_pref(str): the output file will be named [outfile_pref][Property]_comparison.dat (if make_file=True)
                       e.g.: with outfile_pref = 'MLIP-', for the energy the name would be: MLIP-Energy_comparison.dat 
    units(list): dictionary with key-value pairs like prop-unit with prop in ['energy', 'forces', 'stress']
                 and value being a string with the unit to print for the respective property. If None, the
                 respective units will be eV/at, eV/Angs and GPa.

    Return:
    errs(list): [rmse, mae, R2] 
    '''
    if make_file == True:
        dir = os.path.abspath(dir)
        if not dir.endswith('/'):
            dir = dir + '/'
            
    if isinstance(props, str):
        props = [props]
        
    if not all([x in ['all', 'energy', 'forces', 'stress'] for x in props]):
        raise ValueError("Please give a value or a list of values chosen from ['energy', 'forces', 'stress', 'all']")
    
    if not isinstance(props, list):
        props = [props]
    
    if props == 'all' or (isinstance(props, list) and 'all' in props):
        props =  ['energy', 'forces', 'stress']
    
    if units == None:
        units = dict(energy='eV/at', forces='eV/Angst', stress='GPa')
        
    prop_numbs = dict(energy = 0, forces = 1, stress = 2)
    
    # Retrieve the data
    ext1 = [x.flatten() for x in extract_prop(file1)]
    ext2 = [x.flatten() for x in extract_prop(file2)]
    
    # Compute errors and write data on files
    errs = dict()
    for prop in props:
        i = prop_numbs[prop]
        filename = f'{dir}{outfile_pref}{cap_first(prop)}_comparison.dat'
        mae2 = mae(ext1[i], ext2[i])
        rmse2 = rmse(ext1[i], ext2[i])
        R22 = R2(ext1[i], ext2[i])
        errs[prop] = [rmse2, mae2, R22]
        
        if make_file == True:
            text = f'# rmse: {rmse2:.5f} {units[prop]},    mae: {mae2:.5f} {units[prop]}    R2: {R22:.5f}\n'
            text += f'#  True {low_first(prop)}           Predicted {low_first(prop)}\n'
            for x, y in zip(ext1[i], ext2[i]):
                text += f'{x:.20f}  {y:.20f}\n'
            with open(filename, 'w') as fl:
                fl.write(text)
    return errs


    
def train_pot(mlip_bin, init_path, train_set_path, dir, params, mpirun=''):
    '''
    Function to train the MTP model
    Arguments:
    mpirun(str): command for mpi or similar (e.g. 'mpirun')
    mlip_bin(str): path to the MTP binary
    init_path(str): path to the initialization file (.mtp) for MTP
    train_set_path(str): path to the training set (.cfg)
    dir(str): directory where to save everything
    params(dict): dictionary containing the flags to use; these are the possibilities:
        ene_weight(float): weight of energies in the fitting. Default=1
        for_weight(float): weight of forces in the fitting. Default=0.01
        str_weight(float): weight of stresses in the fitting. Default=0.001
        sc_b_for(float): if >0 then configurations near equilibrium (with roughtly force < <double>)
                  get more weight. Default=0
        val_cfg(str):  filename with configuration to validate
        max_iter(int): maximal number of iterations. Default=1000
        cur_pot_n(str): if not empty, save potential on each iteration with name = cur_pot_n.
        tr_pot_n(str): filename for trained potential. Default=Trained.mtp_
        bgfs_tol(float): stop if error dropped by a factor smaller than this over 50 BFGS iterations. 
                         Default=1e-3
        weighting(str): how to weight configuration wtih different sizes relative to each other. 
                   Default=vibrations. Other=molecules, structures.
        init_par(str): how to initialize parameters if a potential was not pre-fitted. Default is random.
                  Other is same - this is when interaction of all species is the same
                  (more accurate fit, but longer optimization)
        skip_preinit(bool): skip the 75 iterations done when parameters are not given
        up_mindist(bool): updating the mindist parameter with actual minimal interatomic distance in
                          the training set
    '''
    
    def get_flags(params):
        flags = dict(ene_weight = '--energy-weight',
                     for_weight = '--force-weight',
                     str_weight = '--stress-weight',
                     sc_b_for = '--scale-by-force',
                     val_cfg = '--valid_cfgs',
                     max_iter = '--max-iter',
                     cur_pot_n = '--curr-pot-name',
                     tr_pot_n = '--trained-pot-name',
                     bgfs_tol = '--bgfs-conv-tol',
                     weighting = '--weighting',
                     init_par = '--init-params',
                     skip_preinit = '--skip-preinit',
                     up_mindist = '--update-mindist')

        cmd = ''
        for par in list(params.keys()):
            if par == 'skip_preinit':
                if params[par] == True:
                    cmd = f'{cmd} {flags[par]}'
                continue
            elif par == 'up_mindist':
                if params[par] == True:
                    cmd = f'{cmd} {flags[par]}'
                continue
            elif par in list(flags.keys()):
                cmd = f'{cmd} {flags[par]}={params[par]}'         
        return cmd
    
    
    
    dir = os.path.abspath(dir)
    if not dir.endswith('/'):
        dir += '/'
    
    if 'tr_pot_n' not in list(params.keys()):
        params['tr_pot_n'] = 'pot.mtp'
    
    flags = get_flags(params)
    cmd = f'{mpirun} {mlip_bin} train {init_path} {train_set_path} {flags}'
    print(cmd)
    log_path = f'{dir}log_train'
    err_path = f'{dir}err_train'
    with open(log_path, 'w') as log, open(err_path, 'w') as err:
        run(cmd.split(), cwd=dir, stdout=log, stderr=err)
        
def train_pot_from_ase(mlip_bin, init_path, train_set, dir, params, mpirun=''):
    '''
    Function to train the MTP model
    Arguments:
    mpirun(str): command for mpi or similar (e.g. 'mpirun')
    mlip_bin(str): path to the MTP binary
    init_path(str): path to the initialization file (.mtp) for MTP
    train_set(list, ase.atoms.Atoms): list of ase Atoms objects; energy, forces and stresses must have been computed and stored in each Atoms object
    dir(str): directory where to save everything
    params(dict): dictionary containing the flags to use; these are the possibilities:
        ene_weight(float): weight of energies in the fitting. Default=1
        for_weight(float): weight of forces in the fitting. Default=0.01
        str_weight(float): weight of stresses in the fitting. Default=0.001
        sc_b_for(float): if >0 then configurations near equilibrium (with roughtly force < <double>)
                  get more weight. Default=0
        val_cfg(str):  filename with configuration to validate
        max_iter(int): maximal number of iterations. Default=1000
        cur_pot_n(str): if not empty, save potential on each iteration with name = cur_pot_n.
        tr_pot_n(str): filename for trained potential. Default=Trained.mtp_
        bgfs_tol(float): stop if error dropped by a factor smaller than this over 50 BFGS iterations. 
                         Default=1e-3
        weighting(str): how to weight configuration wtih different sizes relative to each other. 
                   Default=vibrations. Other=molecules, structures.
        init_par(str): how to initialize parameters if a potential was not pre-fitted. Default is random.
                  Other is same - this is when interaction of all species is the same
                  (more accurate fit, but longer optimization)
        skip_preinit(bool): skip the 75 iterations done when parameters are not given
        up_mindist(bool): updating the mindist parameter with actual minimal interatomic distance in
                          the training set
    '''
    cfg_path = Path(dir).joinpath('TrainSet.cfg')
    conv_ase_to_mlip2(atoms=train_set,
                      out_path=cfg_path,
                      props=True)
    
    train_pot(mlip_bin=mlip_bin,
              init_path=init_path,
              train_set_path=cfg_path, 
              dir=dir,
              params=params,
              mpirun=mpirun)

def pot_from_ini(fpath):
    with open(fpath, 'r') as fl:
        lines = fl.readlines()
    for line in lines:
        if 'mtp-filename' in line:
            pot = line.split()[1]
            return pot
    return ''

def pot_from_pair_style(ps):
    if 'mlip' in ps:
        return pot_from_ini(ps.split()[1])
        

def calc_efs(mlip_bin, mpirun='', confs_path='in.cfg', pot_path='pot.mtp', out_path='./out.cfg', dir='./'):
    '''
    Function to calculate energies, forces, and stresses for the configurations in in.cfg with
    pot.mtp, writing the result to out.cfg.
    Argments:
    mlip_bin(str): path to the mlip binary file
    mpirun(str): whatever command to parallelise; by default 'mpirun'
    confs_path(str): path to the file containing the configurations on which compute efs (.cfg)
    pot_path(str): path to the potential file (.mtp)
    out_path(str): path of the output file (.cfg)
    dir(str): directory where to calculate efs
    '''
    
    if dir == None:
        raise ValueError('Please specify a directory!')
    dir = os.path.abspath(dir)
    if not dir.endswith('/'):
        dir += '/'
    
    cmd = f'{mpirun} {mlip_bin} calc-efs {pot_path} {confs_path} {out_path}'
    log_path = f'{dir}log_calc_efs'
    err_path = f'{dir}err_calc_efs'
    with open(log_path, 'w') as log, open(err_path, 'w') as err:
        print(cmd)
        run(cmd.split(), cwd=dir, stdout=log, stderr=err)
        

def calc_efs_from_ase(mlip_bin, atoms, mpirun='', pot_path='pot.mtp', cfg_files=False, out_path='./out.cfg', dir='./', write_conf=False, outconf_name=None):
    '''
    Function to calculate energies, forces, and stresses for the configurations in an ASE trajectory with
    pot.mtp, writing the result into the same trajectory (plus out.cfg, if wanted).
    Arguments:
    mlip_bin(str): path to the mlip binary file
    confs_path(str): path to the file containing the configurations on which compute efs (.cfg)
    pot_path(str): path to the potential file (.mtp)
    cfg_files(bool): False = the cfg files used to interface to MTP are deleted 
    out_path(str): path of the cfg output file (.cfg) (relevant only if cfg_file = True)
    dir(str): directory where to calculate efs
    write_conf(bool): True = write the trajectory into a file
    outconf_name(str): name of the file where the trajectory is writetn (only with write_conf=True); by default overwrites.
    '''
    
    
    dir = path(dir)
    # first we need to convert ASE to cfg
    cfg_traj = f'{dir}in.cfg'
    conv_ase_to_mlip2(atoms, cfg_traj, props=False)
    
    # compute the properties
    calc_efs(mlip_bin, mpirun=mpirun, confs_path=cfg_traj, pot_path=pot_path, out_path=f'{dir}{out_path}', dir=dir)
    
    # extract the properties from the results
    energy, forces, stress = extract_prop(f'{dir}{out_path}')
    
    # for each configuration create the SinglePoint calculator and assign it, then "compute" the properties
    for i, atom in enumerate(atoms):
        calc = SinglePointCalculator(atom, energy=energy[i], forces=forces[i], stress=stress[i])
        atom.calc = calc
        atom.get_potential_energy()
    
    if write_conf == True:
        if outconf_name is None:
            outconf_name = f'confs.traj'
        write(f'{dir}{outconf_name}', atoms)
    
    if cfg_files == False:
        os.system(f'rm {cfg_traj} {dir}{out_path}')
        
    return atoms
   
    
    
def find_min_dist(trajectory):
    nconfs = len(trajectory)
    mindist = []
    for iconf, conf in enumerate(trajectory):
        pos = conf.get_positions()
        dist = np.array([])
        for iat1, at1 in enumerate(pos):
            dd = np.linalg.norm(pos[iat1+1:] - at1, axis=1)
            dist = np.concatenate((dist, dd))
        mindist.append(dist.min())
    return min(mindist)

