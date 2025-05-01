import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
import os
import argparse
from copy import deepcopy as cp

import sys

from .utils import data_reader, cap_first, repeat, warn, from_list_to_text, mae, rmse, R2, low_first, path, flatten
from .Graphics_matplotlib import Histogram



from ase.io import read, write
from ase import Atoms
import ase.atoms
from ase.calculators.singlepoint import SinglePointCalculator

from numbers import Number

from subprocess import run



def plot_correlations(dtset_ind, ind, dir='', offsets=None, save=False, units=None):
    '''
    This function plot the correlation graphs for energy, forces and stress,
    Parameters:
    dtset_ind (int): 0 for test set, 1 for training set
    ind (int): 0 for energy, 1 for forces, 2 for stress
    dir (str): path of the directory where the comparison files are
    
    '''
    if dir is not None:
        dir = Path(dir)
    dtset_used = ['Test', 'MLIP'] # MLIP corresponds to using the training set
    #dtset_ind = 0
    # order: 0=energy, 1=forces, 2=stress
    names = ['energy', 'forces', 'stress']
    if units == None:
        units = ['eV/at', 'ev/$\mathrm{\AA}$', 'eV/$\mathrm{\AA}^2$']
    file_names = [dir.joinpath(f'{dtset_used[dtset_ind]}-{cap_first(x)}_comparison.dat') for x in names]
    if offsets is None:
        #offsets = [35, 1.8, 1] # negative offset of the y position of the text box 
        offsets = [0, 0, 0]
    #ind = 1

    data = data_reader(file_names[ind].absolute(), separator="", skip_lines=0, frmt="str")

    rmse = data[0][2] 
    mae = data[0][5]
    #print(f"rmse {rmse}, mae {mae}")
    #print(file_names[ind])
    #print(data[0])
    

    R2_v = R2(x_data, y_data)
    txt = f"rmse: {rmse} {units[ind]}\nmae: {mae} {units[ind]}\nR$^2$: {str(round(R2_v, 4))}"
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
        fig_dir = Path('Figures')
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir.joinpath(f'{fname}.png').absolute(), format='png', dpi=600, bbox_inches='tight')
        plt.savefig(fig_dir.joinpath(f'{fname}.svg').absolute(), format='svg')

        

def conv_ase_to_mlip2(atoms, out_path, props=True):
    '''
    Convert a trajectory file of ASE into a .cfg file for mlip-2
    Arguments:
    atoms(list): list of the Atoms objects (configurations)
    out_path(str): path to the output file (must include the extension .cfg) 
    props(bool): if True energy, stress and forces will be copied too (they must have been computed for each configuration in
                 in the trajectory!); if False no property will be copied.
    '''
    out_path = Path(out_path)
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
    with open(out_path.absolute(), 'w') as fl:
        fl.write(text)
    #print(f'File printed to {out_path}')



def check_sets(bins, trainset_name='TrainSet.traj', testset_name='TestSet.traj', save=False):
    '''
    Function to check the distribution of the temperatures among between the train and test set.
    Arguments:
    bins(numpy array): array with the temperatures to consider
    trainset_name(str): path to the train set
    testset_name(str): path to the test set
    save(bool): save the figures
    '''
    if trainset_name is not None:
        trainset_name = Path(trainset_name)
    if testset_name is not None:
        testset_name = Path(testset_name)
   
    train = read(trainset_name.absolute(), index=":")
    test = read(testset_name.absolute(), index=":")

    t_train = [x.get_temperature() for x in train]
    t_test = [x.get_temperature() for x in test]
    binses = binses

    hist_train = Histogram(t_train)
    hist_train.histofy(mode='n_bins', nbins=len(binses), bins=binses, normalized=True)
    hist_train.make_bars(plot=False)
    hist_train.plot_bars(save=save, fname=Path('Train_sets.png').absolute(), format='png', dpi=600, bbox_inches='tight')

    hist_test = Histogram(t_test)
    hist_test.histofy(mode='custom_centers', bins=binses, normalized=True)
    hist_test.make_bars(plot=False)
    hist_test.plot_bars(save=save, fname=Path('Test_sets.png').absolute(), format='png', dpi=600, bbox_inches='tight')

    
    
    
    
def make_mtp_file(sp_count, mind, maxd, rad_bas_sz, rad_bas_type='RBChebyshev', lev=8, mtps_dir=None, wdir='./', out_name='init.mtp'):
    '''
    Function to create the .mtp file necessary for the training
    Args:
    sp_count(str): species_count; how many elements
    mind(float): min_dist; minimum distance between atoms
    maxd(float): max_dist; maximum distance between atoms
    rad_bas_sz(int): radial_basis_size; size of the radial basis
    rad_bas_type(str): radial_basis_type; set of radial basis (default=RBChebyshev)
    lev(int): MTP level; one of [2, 4,  6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    mtps_dir(str): directory where the untrained .mtp files are stored for all levels
    wdir(str): path to the working directory (where the new .mtp file will be saved)
    out_name(str): name of the .mtp file (must include extension) (only name, no path)
    '''
    
    if mtps_dir is None:
        mtps_dir = Path('./')
        
    if wdir is not None:
        wdir = Path(wdir)

    lev = int(lev)
    src_name = f'{lev:0>2d}.mtp'
    src_path = mtps_dir.joinpath(src_name)
    
    with open(src_path.absolute(), 'r') as fl:
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
    
    if filepath is not None:
        filepath = Path(filepath)
    assert any([structures != None, filepath != None]), f"Either structure or filepath must be given!"
    assert not all([structures != None, filepath != None]), f"Either structure or filepath can be given!"
    if structures != None:
        if isinstance(structures, Atoms):
            structures = [structures]
        elif isinstance(structures, list):
            assert all([isinstance(x, Atoms) for x in structures]), \
                   f"Some element of structures is not an ase.atoms.Atoms object!"
        else: 
            raise TypeError('The structures argument passed must be either an Atom object (or a list of Atom objects) or a .cfg file!')
        return extract_prop_from_ase(structures)
    else:
        return extract_prop_from_cfg(filepath.absolute())

        
def extract_prop_from_ase(structures):
    '''Function to extract energy (per atom), forces and stress from an ase trajectory
    
    Parameters
    ----------
    structures: ase.atoms.Atoms or list of ase.atoms.Atoms
       trajectory 
        
    Returns
    -------
    energy: list fof floats
            energy PER ATOM of each configuration (in eV/atom)
    forces: list of 2D np.arrays of floats
            forces with shape nconfs x natoms x 3 (in eV/Angst)
    stress: list of 2D np.arrays of floats
            stress tensor for each configuration (in eV/Angst^2)
        
    Notes
    -----
    The convention for the stress is that the stress tensor element are:
    sigma = + dE/dn (n=strain) (note the sign!)
    
    '''
    if isinstance(structures, Atoms):
        structures = [structures]
    energy = [x.get_total_energy()/len(x) for x in structures]
    forces = [x.get_forces() for x in structures]
    stress = [x.get_stress() for x in structures]
    
    return energy, forces, stress
    


def extract_prop_from_cfg(filepath):
    '''Function to extract energy (per atom), forces and stress from a set of configurations contained in
    a single .cfg file.
    
    Parameters
    ----------
    filepath: str
        path to the .cfg file containing the configurations
        
    Returns
    -------
    energy: list of float
            energy PER ATOM of each configuration (in eV/atom)
    forces: list of 2D numpy.array of float
            forces with shape confs x natoms x 3 (in eV/Angst)
    stress: list of 2D numpy.array of float
            stress tensor for each configuration (in eV/Angst^2)
        
    Notes
    -----
    The convention for the stress is that the stress tensor element are:
    sigma = + dE/dn (n=strain) (note the sign!)
    
    '''
    filepath = Path(filepath)
    with open(filepath.absolute(), 'r') as fl:
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
                natoms = int(lines[i+1])
                curr_forces = np.zeros((natoms, 3))
                curr_stress = np.zeros((6))

            if 'Supercell' in line:
                cell = []
                cell.append( [float(x) for x in lines[i+1].split()] )
                cell.append( [float(x) for x in lines[i+2].split()] )
                cell.append( [float(x) for x in lines[i+3].split()] )
                next(iterator)
                next(iterator)
                next(iterator)
                
            if "AtomData" in line:
                for j in range(natoms):
                    curr_forces[j,0] += float(lines[i + 1 + j].split()[5])
                    curr_forces[j,1] += float(lines[i + 1 + j].split()[6])
                    curr_forces[j,2] += float(lines[i + 1 + j].split()[7])

                for k in range(natoms):
                    next(iterator)
                    
            if "Energy" in line:
                energy.append(float(lines[i+1])/natom) # ENERGY PER ATOM!!!!
            
            if "PlusStress" in line:
                at = Atoms(numbers=[1], cell=cell)
                V = at.get_volume()
                curr_stress[0] += (-float(lines[i+1].split()[0])/V) # xx
                curr_stress[1] += (-float(lines[i+1].split()[1])/V) # yy
                curr_stress[2] += (-float(lines[i+1].split()[2])/V) # zz
                curr_stress[3] += (-float(lines[i+1].split()[3])/V) # yz
                curr_stress[4] += (-float(lines[i+1].split()[4])/V) # xz
                curr_stress[5] += (-float(lines[i+1].split()[5])/V) # xy

            forces.append(curr_forces)
            stress.append(curr_stress)
        
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
    units: dict, default: {'energy': 'eV/at', 'forces':'eV/Angs', 'stress':'eV/Angst^2'}
        dictionary with key-value pairs like prop-unit with prop in 
        ['energy', 'forces', 'stress'] and value being a string with the unit
        to print for the respective property. If None, the respective units
        will be eV/at, eV/Angs and GPa

    Returns
    -------
    errs: dict
    dictionary whose key/vlaue elements are property/errors, where property can be 
    {'energy', 'forces', 'stress'}, and errors is a list [rmse, mae, R2]. 
        
    '''
    
    if is_ase1 == True:
        assert (structures1 != None), f"When is_ase1 = True, " \
            + f"structures1 must be given!"
        if isinstance(structures1, Atoms):
            structures1 = [structures1]
    else:
        assert file1 is not None, f"When is_ase1 = False, file1 must be given!"
        file1 = Path(file1)
        assert file1.is_file() == True, f"{file1.absolute()} is not a file!"
        
    if is_ase2 == True:
        assert (structures2 != None), f"When is_ase2 = True, " \
            + f"structures2 must be given!"
        if isinstance(structures2, Atoms):
            structures2 = [structures2]
    else:
        assert file2 is not None, f"When is_ase2 = False, file2 must be given!"
        file2 = Path(file2)
        assert file2.is_file() == True, f"{file2.absolute()} is not a file!"
    
    if make_file == True:
        if dir is not None:
            dir = Path(dir)
                    
    if isinstance(props, str):
        props = [props]
        
    if not all([x in ['all', 'energy', 'forces', 'stress'] for x in props]):
        raise ValueError("Please give a value or a list of values chosen from ['energy', 'forces', 'stress', 'all']")
    
    if not isinstance(props, list):
        props = [props]
    
    if props == 'all' or (isinstance(props, list) and 'all' in props):
        props =  ['energy', 'forces', 'stress']
    
    if units == None:
        units = dict(energy='eV/at', forces='eV/$\mathrm{\AA}$', stress='eV/$\mathrm{\AA}^2$')
        
    prop_numbs = dict(energy = 0, forces = 1, stress = 2)
    
    # Retrieve the data
    if is_ase1 == True:
        ext1 = [flatten(x) for x in extract_prop_from_ase(structures1)]
    else:
        ext1 = [flatten(x) for x in extract_prop_from_cfg(file1)]


    ext1[0] = ext1[0] #* natoms

    if is_ase2 == True:
        ext2 = [flatten(x) for x in extract_prop_from_ase(structures2)]
    else:
        ext2 = [flatten(x) for x in extract_prop_from_cfg(file2)]
    ext2[0] = ext2[0] #* natoms

    assert len(ext1) == len(ext2), f"You gave a different number of "\
        + f"true and ML structures!"

    dir = Path(dir)
    # Compute errors and write data on files
    errs = dict()
    for prop in props:
        i = prop_numbs[prop]
        filename = dir.joinpath(f'{outfile_pref}{cap_first(prop)}_comparison.dat')
        mae2 = mae(ext1[i], ext2[i])
        rmse2 = rmse(ext1[i], ext2[i])
        R22 = R2(ext1[i], ext2[i])
        errs[prop] = [rmse2, mae2, R22]
        
        if make_file == True:
            print(f'printing in {filename.absolute()}')
            text = f'# rmse: {rmse2:.5f} {units[prop]},    mae: {mae2:.5f} {units[prop]}    R2: {R22:.5f}\n'
            text += f'#  True {low_first(prop)}           Predicted {low_first(prop)}\n'
            for x, y in zip(ext1[i], ext2[i]):
                text += f'{x:.20f}  {y:.20f}\n'
            with open(filename.absolute(), 'w') as fl:
                fl.write(text)
    return errs

def set_level_to_pot_file(trained_pot_file_path, mtp_level):
    fp = Path(trained_pot_file_path)
    assert fp.is_file(), f"{fp.absolute()} is not a regular file!"
    assert isinstance(mtp_level, int) or \
           isinstance(mtp_level, float), f"mtp_level must be an integer!"
    mtp_level = int(mtp_level) # in case it's float
    with open(fp.absolute(), 'r') as fl:
        lines = fl.readlines()
    with open(fp.absolute(), 'w') as fl:
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
                  mpirun='',
                  final_evaluation=False):
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
    mlip_bin = Path(mlip_bin)
    untrained_pot_file_dir = Path(untrained_pot_file_dir)
    train_set_path = Path(train_set_path)
                 
    if 'trained_pot_name' not in list(params.keys()):
        params['trained_pot_name'] = 'pot.mtp'
    flags = get_flags(params)
    make_mtp_file(sp_count=species_count,
                  mind=min_dist,
                  maxd=max_dist,
                  rad_bas_sz=radial_basis_size,
                  rad_bas_type=radial_basis_type, 
                  lev=mtp_level, 
                  mtps_dir=untrained_pot_file_dir.absolute(),
                  wdir=dir.absolute(), 
                  out_name='init.mtp')
    init_path = Path(dir).joinpath('init.mtp')
    cmd = f'{mpirun} {Path(mlip_bin).absolute()} train {init_path.absolute()} {train_set_path.absolute()} {flags}'
    log_path = dir.joinpath('log_train')
    err_path =dir.joinpath('err_train')
    with open(log_path.absolute(), 'w') as log, open(err_path.absolute(), 'w') as err:
        run(cmd.split(), cwd=dir.absolute(), stdout=log, stderr=err)
    trained_pot_file_path = dir.joinpath(params['trained_pot_name']).absolute()
    set_level_to_pot_file(trained_pot_file_path=trained_pot_file_path, mtp_level=mtp_level)
    
    if final_evaluation == True:
        eval_dir = dir.joinpath('evaluation')
        if not eval_dir.is_dir():
            eval_dir.mkdir(parents=True, exist_ok=True)
        calc_efs(mlip_bin.absolute(),
                 mpirun=mpirun, 
                 confs_path=train_set_path.absolute(),
                 pot_path=trained_pot_file_path,
                 out_path=eval_dir.joinpath('ML_dataset.cfg'),
                 dir=eval_dir.absolute())
        
        make_comparison(is_ase1=False,
                        is_ase2=False,
                        structures1=None, 
                        structures2=None, 
                        file1=train_set_path.absolute(),
                        file2=eval_dir.joinpath('ML_dataset.cfg').absolute(),
                        props='all', 
                        make_file=True, 
                        dir=eval_dir,
                        outfile_pref='MLIP-', 
                        units=None)
        
def train_pot(mlip_bin, init_path, train_set_path, dir, params, mpirun='', final_evaluation=False):
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
    mlip_bin = Path(mlip_bin)
    train_set_path = Path(train_set_path) 
    init_path = Path(init_path)
    
    if 'trained_pot_name' not in list(params.keys()):
        params['trained_pot_name'] = 'pot.mtp'
    
    flags = get_flags(params)
    cmd = f'{mpirun} {mlip_bin.absolute()} train {init_path.absolute()} {train_set_path.absolute()} {flags}'
    log_path = dir.joinpath('log_train')
    err_path = dir.joinpath('err_train')
    with open(log_path.absolute(), 'w') as log, open(err_path.absolute(), 'w') as err:
        run(cmd.split(), cwd=dir.absolute(), stdout=log, stderr=err)
    trained_pot_file_path = dir.joinpath(params['trained_pot_name']).absolute()
    set_level_to_pot_file(trained_pot_file_path=trained_pot_file_path, mtp_level=mtp_level)

    if final_evaluation == True:
        eval_dir = dir.joinpath('evaluation')
        if not eval_dir.is_dir():
            eval_dir.mkdir(parents=True, exist_ok=True)
        calc_efs(mlip_bin.absolute(),
                 mpirun=mpirun, 
                 confs_path=train_set_path.absolute(),
                 pot_path=trained_pot_file_path,
                 out_path=eval_dir.joinpath('ML_dataset.cfg'),
                 dir=eval_dir.absolute())
        
        make_comparison(is_ase1=False,
                        is_ase2=False,
                        structures1=None, 
                        structures2=None, 
                        file1=train_set_path.absolute(),
                        file2=eval_dir.joinpath('ML_dataset.cfg').absolute(),
                        props='all', 
                        make_file=True, 
                        dir=eval_dir,
                        outfile_pref='MLIP-', 
                        units=None)
            


def train_pot_from_ase(mlip_bin, init_path, train_set, dir, params, mpirun='', final_evaluation=True):
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
    mlip_bin = Path(mlip_bin)
    init_path = Path(init_path)
    dir = Path(dir)
    cfg_path = dir.joinpath('TrainSet.cfg')
    conv_ase_to_mlip2(atoms=train_set,
                      out_path=cfg_path.absolute(),
                      props=True)
    
    train_pot(mlip_bin=mlip_bin.absolute(),
              init_path=init_path.absolute(),
              train_set_path=cfg_path.absolute(), 
              dir=dir.absolute(),
              params=params,
              mpirun=mpirun,
              final_evaluation=final_evaluation)
    
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
                           mpirun='',
                           final_evaluation=True):
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
    radial_basis_size: int, default=8
            number of basis functions to use for the radial part
    radial_basis_type: {'RBChebyshev', ???}, default='RBChebyshev'
        type of basis functions to use for the radial part
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
    dir = Path(dir)
    mlip_bin = Path(mlip_bin)
    untrained_pot_file_dir = Path(untrained_pot_file_dir)    
    cfg_path = dir.joinpath('TrainSet.cfg')
    
    conv_ase_to_mlip2(atoms=train_set,
                      out_path=cfg_path.absolute(),
                      props=True)
    [at.get_chemical_symbols() for at in train_set]
    species_count = len(set(flatten([at.get_chemical_symbols() for at in train_set])))
    #print('inside train_pot_from_ase_tmp calling for train_pot_tmp')
    train_pot_tmp(mlip_bin=mlip_bin.absolute(),
                  untrained_pot_file_dir=untrained_pot_file_dir.absolute(),
                  mtp_level=mtp_level,
                  species_count=species_count,
                  min_dist=min_dist,
                  max_dist=max_dist,
                  radial_basis_size=radial_basis_size,
                  radial_basis_type=radial_basis_type,
                  train_set_path=cfg_path.absolute(), 
                  dir=dir.absolute(),
                  params=params,
                  mpirun=mpirun,
                  final_evaluation=final_evaluation)
    

def pot_from_ini(fpath):
    with open(Path(fpath).absolute(), 'r') as fl:
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
    
    if dir is None:
        raise ValueError('Please specify a directory!')
        
    dir = Path(dir)
    if confs_path is not None:
        confs_path = Path(confs_path)
    if pot_path is not None:
        pot_path = Path(pot_path)
    if out_path is not None:
        out_path = Path(out_path)
    
    cmd = f'{mpirun} {mlip_bin.absolute()} calc-efs {pot_path.absolute()} {confs_path.absolute()} {out_path.absolute()}'
    log_path = dir.joinpath('log_calc_efs')
    err_path = dir.joinpath('err_calc_efs')
    with open(log_path.absolute(), 'w') as log, open(err_path.absolute(), 'w') as err:
        #print(cmd)
        run(cmd.split(), cwd=dir.absolute(), stdout=log, stderr=err)
        

def calc_efs_from_ase(mlip_bin, 
                      atoms, 
                      mpirun='', 
                      pot_path='pot.mtp', 
                      cfg_files=False, 
                      out_path='./out.cfg',
                      dir='./',
                      write_conf=False, 
                      outconf_name=None):
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
    
    if dir is None:
        raise ValueError('Please specify a directory!')
        
    dir = Path(dir)
    if pot_path is not None:
        pot_path = Path(pot_path)
    if out_path is not None:
        out_path = Path(out_path)
    if mlip_bin is not None:
        mlip_bin = Path(mlip_bin)
    
    # first we need to convert ASE to cfg
    cfg_traj = dir.joinpath('in.cfg')
    atoms = cp(atoms)
    conv_ase_to_mlip2(atoms, cfg_traj.absolute(), props=False)
    
    # compute the properties
    calc_efs(mlip_bin.absolute(), mpirun=mpirun, confs_path=cfg_traj.absolute(), pot_path=pot_path.absolute(), out_path=dir.joinpath(out_path).absolute(), dir=dir.absolute())
    
    # extract the properties from the results
    energy, forces, stress = extract_prop(filepath=dir.joinpath(out_path).absolute()) # energy per atom!!
    # for each configuration create the SinglePoint calculator and assign it, then "compute" the properties
    for i, atom in enumerate(atoms):
        calc = SinglePointCalculator(atom, energy=energy[i]*len(atom), forces=forces[i], stress=stress[i])
        atom.calc = calc
        atom.get_potential_energy()
    
    if write_conf == True:
        if outconf_name is None:
            outconf_name = f'confs.traj'
        write(dir.joinpath(outconf_name).absolute(), atoms)
    
    if cfg_files == False:
        cfg_traj.unlink(missing_ok=True)
        dir.joinpath(out_path).unlink(missing_ok=True)
        
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

        









