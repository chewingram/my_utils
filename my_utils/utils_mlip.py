import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
import os
import importlib.resources
import argparse

import sys

from .utils import data_reader, cap_first, repeat, warn, from_list_to_text, mae, rmse, R2, low_first, path
from .Graphics_matplotlib import Histogram



from ase.io import read, write
from ase import Atoms
import ase.atoms
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
    
    wdir = Path(wdir)

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
    outfile_path = wdir.joinpath(out_name)
    text = from_list_to_text(lines)
    with open(outfile_path.absolute(), 'w') as fl:
        fl.write(text)

def extract_prop(structures=None, filepath=None):
    '''Function to extract properties
    
    It can be either an (list of) ase.atoms.Atoms object(s) or a .cfg file.
    Only one of 'structures' and 'filepath' can be different from None!
    
    Parameters
    ----------
    structures: ase.atoms.Atoms or list of ase.atoms.Atoms
        it can and must be None only if filepath is not None
    filepath: str
        path to the .cfg file; it can and must be None only if structures is not None
    
    
    Returns
    -------
    energy: numpy.array of float
            energy PER ATOM of each configuration (in eV/atom)
    forces: numpy.array of float
            forces with shape confs x natoms x 3 (in eV/Angst)
    stress: numpy.array of float
            stress tensor for each configuration (in eV/Angst^2)
        
    Notes
    -----
    The convention for the stress is that the stress tensor element are:
    sigma = + dE/dn (n=strain) (note the sign!)
    
    
    '''
    
    assert any([structures != None, filepath != None]), f"Either structure or filepath must be given!"
    assert not all([structures != None, filepath != None]), f"Either structure or filepath can be given!"
    if structures != None:
        if isinstance(structures, Atoms):
            structures = [structures]
        elif isinstance(structures, list):
            assert all([isinstance(x, Atoms) for x in structures]), \
                   f"Some element of structures is not an ase.atoms.Atoms object!"
        return extract_prop_from_ase(structures)
    else:
        return extract_prop_from_cfg(filepath)

        
def extract_prop_from_ase(structures):
    '''Function to extract energy (per atom), forces and stress from an ase trajectory
    
    Parameters
    ----------
    structures: ase.atoms.Atoms or list of ase.atoms.Atoms
       trajectory 
        
    Returns
    -------
    energy: numpy.array of float
            energy PER ATOM of each configuration (in eV/atom)
    forces: numpy.array of float
            forces with shape confs x natoms x 3 (in eV/Angst)
    stress: numpy.array of float
            stress tensor for each configuration (in eV/Angst^2)
        
    Notes
    -----
    The convention for the stress is that the stress tensor element are:
    sigma = + dE/dn (n=strain) (note the sign!)
    
    '''
    
    if isinstance(structures, Atoms):
        structures = [structures]
    energy = np.array([x.get_total_energy()/len(x) for x in structures], dtype='float')
    forces = np.array([x.get_forces() for x in structures], dtype='float')
    stress = np.array([x.get_stress() for x in structures], dtype='float')
    
    return energies, forces, stress
    


def extract_prop_from_cfg(filepath):
    '''Function to extract energy (per atom), forces and stress from a set of configurations contained in
    a single .cfg file.
    
    Parameters
    ----------
    filepath: str
        path to the .cfg file containing the configurations
        
    Returns
    -------
    energy: numpy.array of float
            energy PER ATOM of each configuration (in eV/atom)
    forces: numpy.array of float
            forces with shape confs x natoms x 3 (in eV/Angst)
    stress: numpy.array of float
            stress tensor for each configuration (in eV/Angst^2)
        
    Notes
    -----
    The convention for the stress is that the stress tensor element are:
    sigma = + dE/dn (n=strain) (note the sign!)
    
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
    
    
def make_comparison(is_ase1=True,
                    is_ase2=True,
                    structures1=None, 
                    structures2=None, 
                    file1=None,
                    file2=None,
                    props='all', 
                    make_file=False, 
                    dir='',
                    outfile_pref='', 
                    units=None):
    '''Create the comparison files for energy, forces and stress starting from the .cfg files.
    
    Parameters
    ----------
    is_ase1, is_ase2: bool
        - True: an (list of) ase.atoms.Atoms object(s) is expected 
                (inside structures1/structures2) 
        - False: the path to a .cfg files is expected (inside file1/file2)
    structures1: ase.atoms.Atoms or list of ase.atoms.Atoms
        mandatory when is_ase1 = True (ignored otherwise); (list of) ase
        Atoms object(s) with the true values
    structures2: ase.atoms.Atoms or list of ase.atoms.Atoms
        mandatory when is_ase2 = True (ignored otherwise); (list of) ase
        Atoms object(s) with the ML values
    file1: str
        mandatory when is_file1 = True (ignored otherwise); PATH to the
        file with the true values (.cfg)
    file2: str
        mandatory when is_file1 = True (ignored otherwise); PATH to the
        file with the true values (.cfg)
    props: str or list of {'energy', 'forces', 'stress', 'all'}
        if a list is given containing 'all', all three properties will be
        considered, independent on the other elements of the list
    make_file: bool
        - True: create a comparison file
    dir: str
        directory the output file will be saved (if make_file=True)
    outfile_pref: str
        the output file will be named [outfile_pref][Property]_comparison.dat 
        (if make_file=True) e.g.: with outfile_pref = 'MLIP-', for the energy 
        the name would be: MLIP-Energy_comparison.dat 
    units: dict, default: {'energy': 'eV/at', 'forces':'eV/Angs', 'stress':'GPa'}
        dictionary with key-value pairs like prop-unit with prop in 
        ['energy', 'forces', 'stress'] and value being a string with the unit
        to print for the respective property. If None, the respective units
        will be eV/at, eV/Angs and GPa

    Returns
    -------
    errs: list of float
        [rmse, mae, R2] 
        
    '''
    
    if is_ase1 == True:
        assert (structures1 != None), f"When is_ase1 = True, " \
            + f"structures1 must be given!"
        if isinstance(structures1, Atoms):
            structures1 = [structures1]
    else:
        assert file1 != None, f"When is_ase1 = False, file1 must be given!"
        file1 = Path(file1)
        assert file1.is_file() == True, f"{file1.absolute()} is not a file!"
        
    if is_ase2 == True:
        assert (structures2 != None), f"When is_ase2 = True, " \
            + f"structures2 must be given!"
        if isinstance(structures2, Atoms):
            structures2 = [structures2]
    else:
        assert file2 != None, f"When is_ase2 = False, file2 must be given!"
        file2 = Path(file1)
        assert file2.is_file() == True, f"{file2.absolute()} is not a file!"
    
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
    if is_ase1 == True:
        ext1 = [x.flatten() for x in extract_prop_from_ase(structures1)]
    else:
        ext1 = [x.flatten() for x in extract_prop_from_cfg(file1)]
         
    if is_ase2 == True:
        ext2 = [x.flatten() for x in extract_prop_from_ase(structures2)]
    else:
        ext2 = [x.flatten() for x in extract_prop_form_cfg(file2)]

    assert len(ext1) == len(ext2), f"You gave a different number of "\
        + f"true and ML structures!"
        
    
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

def set_level_to_pot_file(trained_pot_file_path, mtp_level):
    fp = Path(trained_pot_file_path)
    assert fp.is_file(), f"{fp.absolute()} is not a regular file!"
    assert isinstance(mtp_level, int) or \
           isinstance(mtp_level, float), f"mtp_level must be an integer!"
    mtp_level = int(mtp_level) # in case it's float
    with open(fp, 'r') as fl:
        lines = fl.readlines()
    with open(fp, 'w') as fl:
        for i, line in enumerate(lines):
            if 'potential_name' in line:
                lines[i] = f'potential_name = MTP_{mtp_level}\n'
        fl.writelines(lines)

def train_pot_tmp(mlip_bin, 
                  untrained_pot_file_dir,
                  mtp_level,
                  min_dist,
                  max_dist,
                  species_count,
                  radial_basis_size,
                  radial_basis_type,
                  train_set_path,
                  dir,
                  params,
                  mpirun=''):
    '''Function to train the MTP model, analogous to train_pot, but the init.mtp file is created by asking the level
    
        Parameters
        ----------
        mpirun: str
            command for mpi or similar (e.g. 'mpirun')
        mlip_bin: str
            path to the MTP binary
        untrained_pot_file_dir: str 
            path to the directory containing the untrained mtp init files (.mtp)
        mtp_level: {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 24, 26, 28}
            level of the mtp model to train
        min_dist: float
            minimum distance between atoms in the system (unit: Angstrom)
        max_dist: float
            cutoff radius for the radial part (unit: Angstrom)
        species_count: int
            number of elements present in the dataset
        radial_basis_size: int, default=8
            number of basis functions to use for the radial part
        radial_basis_type: {'RBChebyshev', ???}, default='RBChebyshev'
            type of basis functions to use for the radial part
        train_set_path: str 
            path to the training set (.cfg)
        dir: str
            path to the directory where to run the training (and save the output)
        params: dict 
            dictionary containing the flags to use; these are the possibilities:
            ene_weight: float, default=1
                weight of energies in the fitting
            for_weight: float, default=0.01
                weight of forces in the fitting
            str_weight: float, default=0.001 
                weight of stresses in the fitting
            sc_b_for: float, default=0
                if >0 then configurations near equilibrium (with roughtly force < 
                <double>) get more weight
            val_cfg: str 
                filename with configuration to validate
            max_iter: int, default=1000
                maximal number of iterations
            cur_pot_n: str
                if not empty, save potential on each iteration with name = cur_pot_n
            trained_pot_name: str, default="Trained.mtp"
                filename for trained potential.
            bfgs_tol: float, default=1e-3
                stop if error dropped by a factor smaller than this over 50 BFGS 
                iterations
            weighting: {'vibrations', 'molecules', 'structures'}, default=vibrations 
                how to weight configuration wtih different sizes relative to each 
                other
            init_par: {'random', 'same'}, default='random'
                how to initialize parameters if a potential was not pre-fitted;
                - random: random initialization
                - same: this is when interaction of all species is the same (more 
                        accurate fit, but longer optimization)
            skip_preinit: bool 
                skip the 75 iterations done when parameters are not given
            up_mindist: bool
                updating the mindist parameter with actual minimal interatomic 
                distance in the training set
    
    '''
    
    
    def get_flags(params):
        flags = dict(ene_weight = '--energy-weight',
                     for_weight = '--force-weight',
                     str_weight = '--stress-weight',
                     sc_b_for = '--scale-by-force',
                     val_cfg = '--valid_cfgs',
                     max_iter = '--max-iter',
                     cur_pot_n = '--curr-pot-name',
                     trained_pot_name = '--trained-pot-name',
                     bfgs_tol = '--bfgs-conv-tol',
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
    
    dir = Path(dir)
    
    if 'trained_pot_name' not in list(params.keys()):
        params['trained_pot_name'] = 'pot.mtp'
    train_set_path = Path(train_set_path)
    flags = get_flags(params)
    make_mtp_file(sp_count=species_count,
                  mind=min_dist,
                  maxd=max_dist,
                  rad_bas_sz=radial_basis_size,
                  rad_bas_type=radial_basis_type, 
                  lev=mtp_level, 
                  mtps_dir=untrained_pot_file_dir,
                  wdir=dir, 
                  out_name='init.mtp')
    init_path = Path(dir).joinpath('init.mtp').absolute()
    cmd = f'{mpirun} {Path(mlip_bin).absolute()} train {init_path.absolute()} {train_set_path.absolute()} {flags}'
    log_path = dir.joinpath('log_train')
    err_path =dir.joinpath('err_train')
    with open(log_path, 'w') as log, open(err_path, 'w') as err:
        run(cmd.split(), cwd=dir, stdout=log, stderr=err)
    set_level_to_pot_file(trained_pot_file_path=dir.joinpath(params['trained_pot_name']).absolute(), mtp_level=mtp_level)    
        
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
        trained_pot_name(str): filename for trained potential. Default=Trained.mtp_
        bfgs_tol(float): stop if error dropped by a factor smaller than this over 50 BFGS iterations. 
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
                     trained_pot_name = '--trained-pot-name',
                     bfgs_tol = '--bfgs-conv-tol',
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
    
    
    
    dir = Path(dir) 
    
    if 'trained_pot_name' not in list(params.keys()):
        params['trained_pot_name'] = 'pot.mtp'
    
    flags = get_flags(params)
    cmd = f'{mpirun} {mlip_bin} train {init_path} {train_set_path} {flags}'
    log_path = f'{dir}log_train'
    err_path = f'{dir}err_train'
    with open(log_path, 'w') as log, open(err_path, 'w') as err:
        run(cmd.split(), cwd=dir, stdout=log, stderr=err)
    set_level_to_pot_file(trained_pot_file_path=dir.joinpath(params['trained_pot_name']), mtp_level=mtp_level)


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
        trained_pot_name(str): filename for trained potential. Default=Trained.mtp_
        bfgs_tol(float): stop if error dropped by a factor smaller than this over 50 BFGS iterations. 
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
    
def train_pot_from_ase_tmp(mlip_bin,
                           untrained_pot_file_dir,
                           mtp_level,
                           min_dist,
                           max_dist,
                           radial_basis_size,
                           radial_basis_type,
                           train_set,
                           dir,
                           params,
                           mpirun=''):
    '''Function to train the MTP model
    
    Parameters
    ----------
    mpirun: str
        command for mpi or similar (e.g. 'mpirun')
    mlip_bin: str
        path to the MTP binary
    untrained_pot_file_dir: str 
        path to the directory containing the untrained mtp init files (.mtp)
    mtp_level: {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 24, 26, 28}
        level of the mtp model to train
    min_dist: float
            minimum distance between atoms in the system (unit: Angstrom)
    max_dist: float
            cutoff radius for the radial part (unit: Angstrom)
    train_set: list, ase.atoms.Atoms 
        list of ase Atoms objects; energy, forces and stresses must have been 
        computed and stored in each Atoms object
    dir: str
        path to the directory where to run the training (and save the output)
    params: dict 
        dictionary containing the flags to use; these are the possibilities:
        ene_weight: float, default=1
            weight of energies in the fitting
        for_weight: float, default=0.01
            weight of forces in the fitting
        str_weight: float, default=0.001 
            weight of stresses in the fitting
        sc_b_for: float, default=0
            if >0 then configurations near equilibrium (with roughtly force < 
            <double>) get more weight
        val_cfg: str 
            filename with configuration to validate
        max_iter: int, default=1000
            maximal number of iterations
        cur_pot_n: str
            if not empty, save potential on each iteration with name = cur_pot_n
        trained_pot_name: str, default="Trained.mtp"
            filename for trained potential.
        bfgs_tol: float, default=1e-3
            stop if error dropped by a factor smaller than this over 50 BFGS 
            iterations
        weighting: {'vibrations', 'molecules', 'structures'}, default=vibrations 
            how to weight configuration wtih different sizes relative to each 
            other
        init_par: {'random', 'same'}, default='random'
            how to initialize parameters if a potential was not pre-fitted;
            - random: random initialization
            - same: this is when interaction of all species is the same (more 
                    accurate fit, but longer optimization)
        skip_preinit: bool 
            skip the 75 iterations done when parameters are not given
        up_mindist: bool
            updating the mindist parameter with actual minimal interatomic 
            distance in the training set
    '''
    cfg_path = Path(dir).joinpath('TrainSet.cfg')
    conv_ase_to_mlip2(atoms=train_set,
                      out_path=cfg_path,
                      props=True)
    species_count = len(list(set(np.array([x.get_chemical_symbols() for  x in train_set]).flatten())))
    print('inside train_pot_from_ase_tmp calling for train_pot_tmp')
    train_pot_tmp(mlip_bin=mlip_bin,
                  untrained_pot_file_dir=untrained_pot_file_dir,
                  mtp_level=mtp_level,
                  species_count=species_count,
                  min_dist=min_dist,
                  max_dist=max_dist,
                  radial_basis_size=radial_basis_size,
                  radial_basis_type=radial_basis_type,
                  train_set_path=cfg_path.absolute(), 
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
    
    Parameters
    ---------
    mlip_bin: str
        path to the mlip binary file
    confs_path: str
        path to the file containing the configurations on which compute efs
        (.cfg)
    pot_path: str
         path to the potential file (.mtp)
    cfg_files: bool
         - False = the cfg files used to interface to MTP are deleted 
    out_path: str
         path of the cfg output file (.cfg) (relevant only if cfg_file = 
         True)
    dir: str
         directory where to calculate efs
    write_conf: bool
         - True = write the trajectory into a file
    outconf_name: str
        name of the file where the trajectory is written (only with 
        write_conf=True); by default overwrites.
        
    Returns
    -------
    atoms: list of ase.atoms.Atoms
        trajectory containing the same input structures with their 
        calculated properties stored

    '''
    
    
    dir = path(dir)
    # first we need to convert ASE to cfg
    cfg_traj = f'{dir}in.cfg'
    conv_ase_to_mlip2(atoms, cfg_traj, props=False)
    
    # compute the properties
    calc_efs(mlip_bin, mpirun=mpirun, confs_path=cfg_traj, pot_path=pot_path, out_path=f'{dir}{out_path}', dir=dir)
    
    # extract the properties from the results
    energy, forces, stress = extract_prop(filepath=f'{dir}{out_path}')

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


def make_ini_for_lammps(pot_file_path, out_file_path):
    '''Function to create the ini file for lammps
    
    Parameters
    ----------

    pot_file_path: str
        path to the .mtp file, that will be printed in the
        ini file for lammps
    out_file_path: str
        path of the ini file to generate

    '''
    pot_file_path = Path(pot_file_path)
    txt = f'mtp-filename {pot_file_path.absolute()}\n'
    txt += f"select FALSE"
    out_file_path = Path(out_file_path)
    with open(out_file_path.absolute(), 'w') as fl:
        fl.writelines(txt)

        
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
    
    parser.add_argument("mode", type=str, default='interactive', help="Mode of using this tool:\n\t- interactive: the input will be asked to the user\n\t- from_file: the input will be extracted from a file")
    parser.add_argument("--fpath", type=str, default=None, help="path to the file containing the instructions")
    
    args = parser.parse_args()
    
    mode = args.mode
    fpath = Path(args.fpath).absolute()
    
    if mode not in ['interactive', 'from_file']:
        raise ValueError('The parameter "mode" must be either "interactive" or "from_file"')
    if mode == 'from_file' and fpath == None:
        raise ValueError('When "mode" = "from_file" a file path must be passed as "fpath"')

        
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
        mode = input("Choose mode\n1 - Homogeneous custom range\n2 - Custom list of temperatures\n")

        if str(mode) == '1':
            input_pars['mode'] = '1'
            input_pars['T1'] = input("T start (K); integer:\n")
            input_pars['T2'] = input("T stop (K); integer (excluded):\n")
            input_pars['step'] = input("step (K); integer:\n")
        elif str(mode) == '2':
            input_pars['mode'] = '2'
            ntemps = input("How many temperatures?\n")
            temps = []
            for i in range(int(ntemps)):
                temp = input(f"Give the temperature n. {i+1}\n")
                temps.append(temp)
            input_pars['temps'] = temps
        else:
            print("You are not the smart one in your  family, are you?")
            
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
        input_pars = convert_input_pars(input_pars)
        
        keys = list(input_pars.keys())
        values = list(input_pars.values())
        for i in range(len(values)):
            print(f'{keys[i]} ---> {values[i]}')
        return input_pars

    def convert_input_pars(input_pars):
        input_pars['wdir'] = Path(input_pars['wdir'][0])
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
                

        return input_pars

    ################

    if mode == 'interactive':
        input_pars = ask_input()
    elif mode == 'from_file':
        input_pars = get_input_from_file(fpath)
        
    root_dir = input_pars['wdir']
    scripts_dir = root_dir.joinpath('scripts_md/')
    os.system(f'mkdir -p {scripts_dir.absolute()}')
      
    scripts_to_copy_dir = Path(__file__).parent.joinpath('data/make_md')
    file_to_copy_names = ['RunNPT_instance.py', 'RunNVT.py', 'LaunchNPT_instances.py', 'nvt_job.sh', 'npt_job.sh']
    for f in file_to_copy_names:
        os.system(f"cp {scripts_to_copy_dir.joinpath(f)} {scripts_dir.absolute()}")
    
    RunNPT_path = scripts_dir.joinpath('RunNPT_instance.py')
    RunNVT_path = scripts_dir.joinpath('RunNVT.py')
        
    matrix = input_pars['matrix']
    
    for i, filepath in enumerate([RunNPT_path, RunNVT_path]):
        tokens = np.zeros(10)
       # filepath = f'{scripts_dir}{file}'
        newlines = []
        with open(filepath.absolute(), 'r') as fl:
            lines = fl.readlines()
        for line in lines:
            if 'rootdir = ' in line:
                newlines.append(f'rootdir = \'{root_dir}\'\n')
            elif 'nproc' in line and tokens[1] == 0:
                newlines.append(f"nproc = {input_pars['ncores'][i]}\n")
                tokens[1] = 1
            elif 'nsteps' in line and tokens[2] == 0:
                newlines.append(f"nsteps = {input_pars['nsteps'][i]}\n")
                tokens[2] = 1
            elif 'loginterval' in line and tokens[3] == 0:
                newlines.append(f"loginterval = {input_pars['loginterval'][i]}\n")
                tokens[3] = 1
            elif 'nthrow' in line and tokens[4] == 0:
                newlines.append(f"nthrow = {input_pars['nthrow'][i]}\n")
                tokens[4] = 1
            elif 'dt' in line and tokens[5] == 0:
                newlines.append(f"dt = {input_pars['dt'][i]}\n")
                tokens[5] = 1
            elif 'iso' in line and tokens[6] == 0:
                newlines.append(f"iso = {input_pars['iso']}\n")
                tokens[6] = 1
            elif 'mult_mat' in line and tokens[7] == 0:
                newlines.append(f"mult_mat = np.array([[{matrix[0,0]}, {matrix[0,1]}, {matrix[0,2]}], [{matrix[1,0]}, {matrix[1,1]}, {matrix[1,2]}], [{matrix[2,0]}, {matrix[2,1]}, {matrix[2,2]}]])\n")
                tokens[7] = 1
            elif 'refine =' in line and tokens[8] == 0:
                newlines.append(f"refine = {str(input_pars['refine'])}\n")
                tokens[8] = 1
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
                newlines.append(f"#SBATCH --time={input_pars['time']}\n")
            else:
                newlines.append(line)
        with open(filepath.absolute(), 'w') as fl:
            fl.writelines(newlines)

    LaunchNPT_instances_path = scripts_dir.joinpath('LaunchNPT_instances.py')
    newlines = []
    with open(LaunchNPT_instances_path.absolute(), 'r') as fl:
        lines = fl.readlines()
    for line in lines:
        if 'rootdir =' in line:
            newlines.append(f'rootdir = \'{root_dir}\'\n')
        elif 'ninstances =' in line:
            newlines.append(f"ninstances = {input_pars['ninstances']}\n")
        else:
            newlines.append(line)
    with open(LaunchNPT_instances_path.absolute(), 'w') as fl:
        fl.writelines(newlines)


    for T in input_pars['temps']:
        temp_dir = root_dir.joinpath(f'T{T}K/')
        os.system(f'mkdir -p {temp_dir.absolute()}')
        os.system(f"ln -s -f {scripts_dir.absolute().joinpath('*')} {temp_dir.absolute()}")



