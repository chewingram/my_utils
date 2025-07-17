import numpy as np
from ase.io import read, write
from ase.build import make_supercell
import sys
from .utils import from_list_of_numbs_to_text, data_reader, ln_s_f
from .utils import print_b, print_bb, print_g, print_gb, print_kb, print_r, print_rb
import os
from subprocess import run
from math import floor, ceil
import h5py
from matplotlib import pyplot as plt
from pathlib import Path
import shutil
from copy import deepcopy as cp
from termcolor import colored
import pickle as pkl
from random import shuffle

binpath = shutil.which('extract_forceconstants')
if binpath is not None:
    g_tdep_bin_directory = Path(binpath).parent.absolute()
else:
    g_tdep_bin_directory = Path('./').absolute()

def get_xcart(path):
    '''
    Function to get the xcart (in Angstrom) of the end set from an .abo file. 
    '''
    path = Path(path)
    natoms = 6
    positions = []
    tok = 0
    lines = data_reader(path.absolute(), separator='')
    for i, line in enumerate(lines):
        if 'END' in line:
            tok = 1
        elif 'xcart' in line:
            if tok == 1:
                at_pos = [float(line[1])* 0.529177249, float(line[2])* 0.529177249, float(line[3])* 0.529177249]
                at_pos = np.array(at_pos)
                positions.append(at_pos)
                for j in range(1, natoms):
                    at_pos = [float(lines[i+j][0])* 0.529177249, float(lines[i+j][1])* 0.529177249, float(lines[i+j][2])* 0.529177249]
                    at_pos = np.array(at_pos)
                    positions.append(at_pos)
    positions = np.array(positions)
    return positions


def get_cell(path):
    '''
    Function to get the cell (in Angstrom) from the end dataset of an .abo file
    '''
    path = Path(path)
    cell = []
    tok = 0
    lines = data_reader(path.absolute(), separator='')
    for i, line in enumerate(lines):
        if 'END' in line:
            tok = 1
        # look for acell
        elif 'acell' in line:
            if tok == 1:
                acell = np.array([float(line[1]), float(line[2]), float(line[3])])
        elif 'rprim' in line:
            if tok == 1:
                rprim1 = np.array([float(line[1]), float(line[2]), float(line[3])])
                rprim2 = np.array([float(lines[i+1][0]), float(lines[i+1][1]), float(lines[i+1][2])])
                rprim3 = np.array([float(lines[i+2][0]), float(lines[i+2][1]), float(lines[i+2][2])])
                rprim = [rprim1* 0.529177249, rprim2* 0.529177249, rprim3* 0.529177249]
                rprim = np.array(rprim)
    rprimd = cp(rprim)
    for i in range(len(rprim)):
        rprimd[i] = rprim[i] * acell[i]
    
    
    return rprimd, acell, rprim

def merge_confs(n_conf, dir, pref='aims_conf', filename='canonical_structures.traj'):
    '''
    Function for merging the structures obtained with canonical_configuration and deleting the old individual configuration files, ignoring the bugged or corrupted ones.
    Arguments:
    n_conf(int): number of configurations to merge
    dir(str): directory where the structure files are where the merged file will be
    pref(str): prefix before the number in the name of the structure files

    '''
    n_len = len(str(int(n_conf))) # number of digits of n_conf
    n_len = 4
    confs = []
    dir = Path(dir)
    
    for i in range(n_conf):
        fname = f'{pref}{i+1:0>{n_len}d}'
        fpath = dir.joinpath(fname)
        try:
            conf = read(fpath.absolute(), format='aims')
            
        except: # e.g. the file is bugged/corrupted AND can't be opened
            continue

        if len(conf) != 0:
            confs.append(cp(conf))
        else: # e.g. the file is openable but empty
            continue

        fpath.unlink(missing_ok=False) # now we can delete the original file
    if len(confs)>0:
        write(dir.joinpath(filename).absolute(), confs, format='traj')
        
#     for i in range(n_conf):
#         #fname = dir.joinpath(f'{pref}{i+1:0>4d}')
        
#         os.system(f'cp {fname.absolute()} /scratch/users/s/l/slongo/Work/ML/MoS2/new_mlacs/sTDEP/debug/')
#         fname.unlink(missing_ok=True)
    return len(confs)

def make_stat(confs, dir):
    '''
    Function to write the infile.stat given an ase trajectory
    Arguments:
    confs(list): ASE trajectory
    dir(str): directory where the infile will be printed
    '''
    dir = Path(dir)
    text = ''
    for i, atoms in enumerate(confs):
        time = 0
        temp = atoms.get_temperature()
        Etot = atoms.get_total_energy()
        Ekin = atoms.get_kinetic_energy()
        Epot = atoms.get_potential_energy()
        stress = atoms.get_stress()
        press = -1/3 * (stress[0] + stress[1] + stress[2])
        line = f'{i+1}\t{time}\t{Etot}\t{Epot}\t{Ekin}\t{temp}\t{press}\t{stress[0]}\t{stress[1]}\t{stress[2]}\t{stress[3]}\t{stress[4]}\t{stress[5]}'
        text += line + '\n'
    with open(dir.joinpath('infile.stat').absolute(), 'w') as fl:
        fl.write(text)

def make_fake_stat(confs, dir, temp):
    '''
    Function to write a fake infile.stat given an ase trajectory. Energy, stress, pressure and timesteps will be set to 0.
    This can be helpful, for example, when using configurations obtained with canonical_configuration.
    Arguments:
    confs(list): ASE trajectory
    dir(str): directory where the infile will be printed
    temp(float): temperature
    '''
    dir = Path(dir)
    text = ''
    for i, atoms in enumerate(confs):
        time = 0
        temp = temp
        Etot = 0
        Ekin = 0
        Epot = 0
        stress = [0, 0, 0 ,0 ,0 ,0]
        press = 0
        line = f'{i+1}\t{time}\t{Etot}\t{Epot}\t{Ekin}\t{temp}\t{press}\t{stress[0]}\t{stress[1]}\t{stress[2]}\t{stress[3]}\t{stress[4]}\t{stress[5]}'
        text += line + '\n'
    with open(dir.joinpath('infile.stat').absolute(), 'w') as fl:
        fl.write(text)

def make_positions(confs, dir):
    '''
    Function to write the infile.positions given an ase trajectory
    Arguments:
    confs(list): ASE trajectory
    dir(str): directory where the infile will be printed
    '''
    dir = Path(dir)
    pos = []
    for atoms in confs:
        pos.extend([[x[0], x[1], x[2]] for x in atoms.get_scaled_positions()])
    
    text = from_list_of_numbs_to_text(pos)
    with open(dir.joinpath('infile.positions').absolute(), 'w') as fl:
        fl.write(text)


def make_forces(confs, dir):
    '''
    Function to write the infile.forces given an ase trajectory
    Arguments:
    confs(list): ASE trajectory
    dir(str): directory where the infile will be printed
    '''
    dir = Path(dir)
    forces = []
    for atoms in confs:
        forces.extend([[x[0], x[1], x[2]] for x in atoms.get_forces()])
    
    text = from_list_of_numbs_to_text(forces)
    with open(dir.joinpath('infile.forces').absolute(), 'w') as fl:
        fl.write(text)

def make_fake_forces(confs, dir):
    '''
    Function to write a fake infile.forces given an ase trajectory. All components of all forces will be set to 0.
    This can be helpful, for example, when using configurations obtained with canonical_configuration.
    Arguments:
    confs(list): ASE trajectory
    dir(str): directory where the infile will be printed
    '''
    dir = Path(dir)
    forces = []
    for atoms in confs:
        for i in range(len(atoms)):
            forces.append([0, 0, 0])
    text = from_list_of_numbs_to_text(forces)
    with open(dir.joinpath('infile.forces').absolute(), 'w') as fl:
        fl.write(text)

def make_meta(confs, dir, timestep=1, temp=None):
    '''
    Function to write the infile.meta given an ase trajectory
    Arguments:
    confs(list): ASE trajectory
    dir(str): directory where the infile will be printed
    temp(float): the temperature to consider. If None, it will be set as the average of all confs.
    '''
    dir = Path(dir)
    if temp is None:
        tmp = np.array([x.get_temperature() for x in confs])
        temp = np.mean(tmp)
    text = ''
    text += str(len(confs[0])) + '\n'
    text += str(len(confs)) + '\n'
    text += '1\n'
    text += str(temp)
    with open(dir.joinpath('infile.meta').absolute(), 'w') as fl:
        fl.write(text)

def make_canonical_configurations(ucell, scell, nconf, temp, quantum, dir, outfile_name, max_freq=False, ifcfile_path=None, pref_bin='', tdep_bin_directory=None):
    '''
    Function to launch tdep canonical-configurations specifying only the number of confs, the temperature and
    the quantum flag. After this, all (non bugged) files are merged in a single ASE traj (outfile_name).
    If canonical-sampling produces some bugged confs, it will be regenerated and saved in the output file.
    Args:
    ucell(ase.Atoms): unit cell
    scell(ase.Atoms): supercell
    nconf(int): number of configurations to generate
    temp(float): temperature
    quantum(bool): True: quantum phonon distribution; False: classical phonon distribution
    dir(str): path to the directory where to do this (note that infiles must be present)
    outfile_name(str): name of the final ASE trajectory (must include extension) containing the configurations.
    tdep_bin_directory(str): path to the directory containing the tdep binaries
    '''
    
    if tdep_bin_directory is None or tdep_bin_directory == '':
        tdpdir = g_tdep_bin_directory
    else:
        tdpdir = Path(tdep_bin_directory) # if it's already a Path(), this won't change anything

    dir = Path(dir)
    
# *1: sometime canonical-configuration creates some bugged configuration file. Then, after the generation,
#     all files are merged using tdp.merge_confs(), which only merges the files that are not bugged.
#     For this reason, the output file can contain less conf than we wanted to generate.
    def merge_trajs(traj1, traj2):
        at1 = read(traj1, index=':')
        at2 = read(traj2, index=':')
        at1.extend(at2)
        return at1
    
    tok = 0
    
    write(dir.joinpath('infile.ucposcar'), ucell, format='vasp')
    write(dir.joinpath('infile.ssposcar'), scell, format='vasp')
    pref_bin = '' # we force this because the canonical_configurations binary is kinda bugged and doesn't work in parallel
    if max_freq != False:
        max_freq = f'--maximum_frequency {max_freq}'
    else:
        max_freq = ''
        if ifcfile_path is None:
            raise TypeError('Since max_freq is False, you must provide ifcfile_path!')
        ln_s_f(Path(ifcfile_path), dir.joinpath('infile.forceconstant'))
            
    while tok < 1:
        if quantum == True:
            cmd = f'{pref_bin} {tdpdir.joinpath("canonical_configuration").absolute()} -of 4 -n {nconf} -t {temp} {max_freq} --quantum'
        elif quantum == False:
            cmd = f'{pref_bin} {tdpdir.joinpath("canonical_configuration").absolute()} -of 4 -n {nconf} -t {temp} {max_freq}'
        
        logpath = dir.joinpath('log_cs')
        errpath = dir.joinpath('err_cs')
        with open(logpath.absolute(), 'w') as log, open(errpath.absolute(), 'w') as err:
            run(cmd.split(), cwd=dir.absolute(), stdout=log, stderr=err)
        # merge the confs available
        n_done = merge_confs(n_conf=nconf, dir=dir.absolute(), filename=outfile_name)

        # if an old mergexd-confs file exists, it must be merged to the new one and then deleted
        if dir.joinpath(f'old_{outfile_name}').exists():
            if n_done>0:
                merged_traj = merge_trajs(dir.joinpath(f'old_{outfile_name}'), dir.joinpath(outfile_name))
                write(dir.joinpath(outfile_name), merged_traj)
                dir.joinpath(f'old_{outfile_name}').unlink(missing_ok=True)
            else:
                dir.joinpath(f'old_{outfile_name}').replace(dir.joinpath(outfile_name))

        # now check that the newly generated confs are as many as we needed; see *1 above
        if n_done == nconf:  # nconf = n. of conf to generate; n_done = n. of non bugged confs actually generated
            tok = 1 # if all is ok, then we can exit the loop
        else:
            # if we need more confs we change name to the merged confs file and go back for another iteration
            # with the remaining number of confs to generate
            if dir.joinpath(outfile_name).exists(): # we do this only if we actually have at least one non bugged
                                                # conf to save before going to the next iteration
                dir.joinpath(outfile_name).replace(dir.joinpath(f'old_{outfile_name}'))
            nconf -= n_done
            print(f'{nconf} out of {nconf + n_done} configurations are bugged; re-generation will be attempted.')

def extract_msd(fname, T, masses, ncells, quantum):
    '''
    Function to extract the mean squared displacement from the outfile.grid_dispersions.hdf5 file dumped
    by phonon_dispersion_relations (TDEP).
    Args
    fname(str): path to the file (including name)
    T(float): temperature at which compute the msd.
    masses(list): list of mass of each atom in the unit cell (with repetitions, if it's the case) in kg
    ncells(int): number of cells used
    Returns
    atomic_msds(numpy array): the msds of each atom in the unit cell (in Ang^2)
    mean_msd(float): the average msd (in Ang^2).
    
    
    '''
        
    def bose_einstein(freq, T):
        return 1. / (np.exp((hbar * freq) / (k * T)) - 1.)
    
    def classical(freq, T):
        return -0.5 + (k * T) / (hbar * freq)

    fname = Path(fname)
    if quantum == True:
        dist_func = bose_einstein
    elif quantum == False:
        dist_func = classical

    hbar = 6.62607015E-34 / (2 * np.pi) # m^2 kg/
    k = 1.380649E-23 # m^2 kg s^-2 K^-1
    THz_Hz = 1E+12 # to convert from THz to Hz
    m_Ang = 1E+10
    prefact1 = hbar / (2 * ncells) * m_Ang**2
    masses = np.repeat(np.array(masses, dtype=float), 3)
    massesm1 = 1./masses

    fl = h5py.File(fname.absolute(), 'r')
    
    evcti = np.array(fl['eigenvectors_im'], dtype=float)
    evctr = np.array(fl['eigenvectors_re'], dtype=float)
    natoms = int(len(evcti[0]) / 3)
    qpts = np.array(fl['qpoints'], dtype=float)
    freqs = np.array(fl['frequencies'], dtype=float) # In Hz * rad  !!!!!
    msds = np.zeros((natoms*3))
    for i_q, q in enumerate(qpts[:]): # qpts
        for i_b in range(3*natoms): # branches
            freq = freqs[i_q][i_b]
            freqm1 = 1./freq
            dist  = dist_func(freq, T)
            prefact2 = prefact1 * freqm1 * (1. + 2. * dist)
            msds += prefact2 * massesm1 * (evcti[i_q][i_b]**2 + evctr[i_q][i_b]**2) # vectorial operations to compute the current branch and
                                                                                    # qpt contribution to the msd of all atoms in a shot
    
    #atomic_msds = np.sum(msds.reshape(natoms, 3), axis=1)
    
    mean_msd = msds.mean()
   
    return mean_msd

def make_anharmonic_free_energy(dir='./', bin_path=None, mpirun='', qgrid=None, thirdorder=False, stochastic=False, quantum=True):
    '''
    Function to launch the anharmonic_free_energy binary of TDEP
    Args:
    dir(str, path): directory where to run the calculation all infiles (ifc, uc, ss, meta, for, pos, meta, stat) must be present
    bin_path(str, path): path to the binary of anharmonic_free_energy
    mpirun(str): string before the binary
    qgrid(str): three integers for the q-pts grid; e.g. 10x10x10
    thirdorder(bool): include thirdorder (infile.force_constant_thirdorder must be in the directory)
    stochastic(bool): True: a stochastic sampling was used; False: a time-evolution sampling (e.g. MD) was used
    quantum(bool): include quantum effects
    '''
    if dir is not None:
        dir = Path(dir) 
    
    if bin_path is None:
        bin_path = g_tdep_bin_directory.joinpath('anharmonic_free_energy')
    else:
        bin_path = Path(bin_path)
        
    qgrid = '' if qgrid is None else f'-qg {qgrid}'
    
    thirdorder = '--thirdorder' if thirdorder == True else ''
    
    stochastic = '--stochastic' if stochastic == True else ''
    
    quantum = '--quantum' if quantum == True else ''
    
    cmd = f'{mpirun} {bin_path.absolute()} {qgrid} {thirdorder} {stochastic} {quantum}'
    logpath = dir.joinpath('log_fe')
    errpath = dir.joinpath('err_fe')
    with open(logpath.absolute(), 'w') as log, open(errpath.absolute(), 'w') as err:
        run(cmd.split(), cwd=dir.absolute(), stdout=log, stderr=err)
    
    # extract the best approximation of the free energy from the outfile
    with open(dir.joinpath('outfile.anharmonic_free_energy'), 'r') as fl:
        lines = fl.readlines()
    fe = float(lines[-1].split()[1]) # in eV/atom
    
    with open(dir.joinpath('infile.meta'), 'r') as fl:
        lines = fl.readlines()
    natoms = int(lines[0].split()[0])
    
    fe = fe * natoms # in eV
    return fe




def run_ifc_and_phonons(root_dir='./Tdep/',
                        temperature=0,
                        sampling_path=None,
                        ucell=None,
                        scell=None,
                        paralrun='',
                        rc2s=[5],
                        rc3s=[-1],
                        lo_to=False,
                        lo_to_file_path=None,
                        cub_root_gds=[26],
                        qgs='default',
                        dos=False,
                        unit=None,
                        nq=None,
                        read_path=False,
                        dump_grid=False,
                        tdep_bin_directory=None):
    '''
    Function to extract the ifcs (up to third order) and phonons, varying rc2, rc3 (for ifcs) and the qpts grid (for phonons)
    The results will be structured according to the following scheme:
    root_dir
        └── ifc
             ├── rc2_N
             │   ├── rc3_n
             │   │   ├── ...
             │   │   └── phonons
             │   │      ├── qg_pxpxp
             │   │      │  ├── ...
             │   │      │  └── outfile.dispersion_relations
             │   │      ├── qg_gxgxg
             │   │      │  ├── ...
             │   │      │  └── outfile.dispersion_relations
             │   │      └── qg_hxhxh              
             │   │           ├── ...
             │   │           └── outfile.dispersion_relations
             │   └── rc3_m
             │       └── ...
             ├── rc2_M
             │   ├── ...             
             │   ... 
             ...
    where {N, M, ...} are the values of rc2, {n, m, ...} are the values of rc3 and {p, g, h, ...} are the values
    inside cub_root_gds (see below).
    
    Args:
    root_dir(str, path): path to the directory where everything will be run; if not existing, it will be created
    temperature(float): temperature
    sampling_path(str, path): path to the ASE trajectory containing the sampled configurations
    ucell(ase.atoms.Atoms): unit cell in ASE format
    scell(ase.atoms.Atoms): supercell in ASE format
    paralrun(str): command to write before the binaries (es. 'mpirun' or 'srun')
    rc2s(list): list of values for the rc2
    rc3s(list): list of values for the rc3
    lo_to(bool): include or not the LO-TO splitting (if True a path must be given for 'lo_to_file_path')
    lo_to_file_path(str, path): path to the infile.lotosplitting file to use for the LO-TO correction
    cub_root_gds(list): list with the cubic root (should be an integer) of the total number of q-points; if 'qgs' does not depend
                        on this variable, then it is only used to name the relevant folder. By default it is truncated to the
                        lower integer and used as number of repetition of the cell along the three unit vectors for the 
                        phonons (e.g. cub_root_gds=16 ==> q grid=16x16x16)
    qgs(list): list of three integers representing the number of repetitions of the cell along the three unit vectors for the 
               phonons. By default the three integers are all the same and equal to 'cub_root_gds'.
    dos(bool): whether to compute the phonon dos
    unit(str): units for the frequency of the phonon dispersion; possible choices: 'thz' (default), 'icm', 'ev'
    nq(int): number of q-point between each high symmetry point
    read_path(bool): whether to read a q-point path or not; if true, something must be given to 'path_file_path'
    path_file_path(str, path): path to the file containing the q-point path for the phonons (in tdep format, see docs); it is 
                               necessary when read_path=True
    dump_grid(bool): whether to dump the q-point grid data from the phonon calculation (see tdep docs).
    tdep_bin_directory(str, path): path to the directory containing the tdep binaries, default: the global tdep_bin_directory variable
    '''
    if root_dir is not None:
        root_dir = Path(root_dir)
        
    if tdep_bin_directory is None or tdep_bin_directory == '':
        tdpdir = g_tdep_bin_directory
    else:
        tdpdir = Path(tdep_bin_directory) # if it's already a Path(), this won't change anything
    
    if sampling_path is not None:
        sampling_path = Path(sampling_path)
        
    # Check parameters for extract_forceconstant
    if lo_to == True:
        polar = '--polar'
        if lo_to_file_path is not None:
            lo_to_file_path = Path(lo_to_file_path)
    else:
        polar = ''

    # Check parameters for phonon_dispersion_relations
    if read_path == True:
        read_path = '-rp'
        pfp = Path(path_file_path)
    else:
        read_path = ''  

    if qgs == 'default':
        qgs = [f'{int(x)} {int(x)} {int(x)}' for x in cub_root_gds]

    if dos == True:
        dos = '--dos'
    else:
        dos = ''

    if dump_grid == True:
        dump_grid = '--dumpgrid'
    else:
        dump_grid = ''

    if nq is None:
        nq = ''
    else:
        nq = f'-nq {nq}'
    if unit is None:
        unit = ''
    else:
        unit = f'--unit {unit}'


    root_dir.mkdir(parents=True, exist_ok=True)

    # SAMPLED CONFS
    traj = read(sampling_path.absolute(), index=':')


    # INFILES
    infiles_dir = root_dir.joinpath('infiles')
    infiles_dir.mkdir(parents=True, exist_ok=True)
    #os.system(f'ln -s -f {root_dir}../T{temperature}K.traj {infiles_dir}/')
    ln_s_f(sampling_path, infiles_dir)
    write(infiles_dir.joinpath('infile.ucposcar').absolute(), ucell, format='vasp')
    write(infiles_dir.joinpath('infile.ssposcar').absolute(), scell, format='vasp')
    make_forces(traj, infiles_dir.absolute())
    make_positions(traj, infiles_dir.absolute())
    make_meta(traj, infiles_dir.absolute(), timestep=1, temp=temperature)
    make_stat(traj, infiles_dir.absolute())


    # START DOING THINGS
    ifc_pref = 'ifc'
    ifc_dir = root_dir.joinpath(ifc_pref)
    ifc_dir.mkdir(parents=True, exist_ok=True)

    for rc2 in rc2s:
        rc2_pref = f'rc2_{rc2}'
        rc2_dir = ifc_dir.joinpath(rc2_pref)
        rc2_dir.mkdir(parents=True, exist_ok=True)

        for rc3 in rc3s:
            rc3_pref = f'rc3_{rc3}'
            rc3_dir = rc2_dir.joinpath(rc3_pref)
            rc3_dir.mkdir(parents=True, exist_ok=True)

            # linking infiles
            ln_s_f(infiles_dir.joinpath("infile.ucposcar"), rc3_dir)
            ln_s_f(infiles_dir.joinpath("infile.ssposcar"), rc3_dir)
            ln_s_f(infiles_dir.joinpath("infile.forces"), rc3_dir)
            ln_s_f(infiles_dir.joinpath("infile.positions"), rc3_dir)
            ln_s_f(infiles_dir.joinpath("infile.meta"), rc3_dir)
            ln_s_f(infiles_dir.joinpath("infile.stat"), rc3_dir)
            
        
            if lo_to == True:
                ln_s_f(lo_to_file_path, rc3_dir)

            cmd = f'{paralrun} {tdpdir.joinpath("extract_forceconstants").absolute()} -rc2 {rc2} -rc3 {rc3} {polar}'
            logpath = rc3_dir.joinpath('log_ifc')
            errpath = rc3_dir.joinpath('err_ifc')
            with open(logpath.absolute(), 'w') as log, open(errpath.absolute(), 'w') as err:
                #print(cmd.split())
                run(cmd.split(), cwd=rc3_dir.absolute(), stdout=log, stderr=err)

            ph_pref = 'phonons'
            ph_dir = rc3_dir.joinpath(ph_pref)
            ph_dir.mkdir(parents=True, exist_ok=True)

            for i, qg in enumerate(qgs):
                qg_pref = f'qg_{cub_root_gds[i]}x{cub_root_gds[i]}x{cub_root_gds[i]}'
                qg_dir = ph_dir.joinpath(qg_pref)
                qg_dir.mkdir(parents=True, exist_ok=True)

                # linking infiles
                ln_s_f(infiles_dir.joinpath("infile.ucposcar"), qg_dir)
                ln_s_f(infiles_dir.joinpath("infile.ssposcar"), qg_dir)
                ln_s_f(infiles_dir.joinpath("infile.forces"), qg_dir)
                ln_s_f(infiles_dir.joinpath("infile.positions"), qg_dir)
                ln_s_f(infiles_dir.joinpath("infile.meta"), qg_dir)
                ln_s_f(infiles_dir.joinpath("infile.stat"), qg_dir)
                ln_s_f(rc3_dir.joinpath("outfile.forceconstant"), qg_dir.joinpath('infile.forceconstant'))
                ln_s_f(rc3_dir.joinpath("outfile.forceconstant_thirdorder"), qg_dir.joinpath('infile.forceconstant_thirdorder'))

                if read_path == '-rp':
                    ln_s_f(pfp.joinpath('infile.qpoints_dispersion'), qg_dir)

                cmd = f'{paralrun} {tdpdir.joinpath("phonon_dispersion_relations").absolute()} -qg {qg} {read_path} {dos} {dump_grid} {unit} {nq}'
                logpath = qg_dir.joinpath('log_ph')
                errpath = qg_dir.joinpath('err_ph')
                with open(logpath.absolute(), 'w') as log, open(errpath.absolute(), 'w') as err:
                    print(cmd.split())
                    run(cmd.split(), cwd=qg_dir.absolute(), stdout=log, stderr=err)

def errs_ifc_phonons(root_dir,
                     rc2s,
                     rc3s,
                     qgs):
    '''
    Function to extract the maximum error by varying each rc2, rc3 and qgs. The folder with results must be 
    strctured according to the following scheme:
    root_dir
        └── ifc
             ├── rc2_N
             │   ├── rc3_n
             │   │   ├── ...
             │   │   └── phonons
             │   │      ├── qg_pxpxp
             │   │      │  ├── ...
             │   │      │  └── outfile.dispersion_relations
             │   │      ├── qg_gxgxg
             │   │      │  ├── ...
             │   │      │  └── outfile.dispersion_relations
             │   │      └── qg_hxhxh              
             │   │           ├── ...
             │   │           └── outfile.dispersion_relations
             │   └── rc3_m
             │       └── ...
             ├── rc2_M
             │   ├── ...             
             │   ... 
             ...
             
    This is how the function run_ifc_and_phonons() arranges the results.
    NOTE! For each combination of the parameters taken from the list below there must exists a relevant calculation. 
    Args:
    root_dir(str, path): path to the root directory (see above)
    rc2s(list): list of values for rc2 to check the convergence for
    rc3s(list): list of values for rc3 to check the convergence for
    qgs(list): list of values for q-point grids (e.g. '16x16x6') to check the convergence for
    '''
    def comp_err(old, new):
        err = []
        for i in range(len(old)):
            if old[i] == 0:
                err.append(0)
            else:
                err.append((new[i]-old[i])/abs(old[i]))
        return np.array(err)

    root_dir = Path(root_dir)
    ifc_pref = 'ifc'
    ifc_dir = root_dir.joinpath(ifc_pref)
    
    res = []
    
    for rc2 in rc2s:
        rc2_pref = f'rc2_{rc2}'
        rc2_dir = ifc_dir.joinpath(rc2_pref)

        for rc3 in rc3s:
            rc3_pref = f'rc3_{rc3}'
            rc3_dir = rc2_dir.joinpath(rc3_pref)
            ph_pref = 'phonons'
            ph_dir = rc3_dir.joinpath('phonons')
            for i, qg in enumerate(qgs):
                qg_pref = f'qg_{qg}'
                qg_dir = ph_dir.joinpath(qg_pref)
                filename = 'outfile.dispersion_relations'
                filepath = qg_dir.joinpath(filename)
                with open(filepath.absolute(), 'r') as fl:
                    lines = fl.readlines()
                lines = np.array([line.split() for line in lines], dtype='float')
                tmp_freqs = lines[:,1:].flatten()
                res_to_store = [rc2, rc3, qg, tmp_freqs]
                res.append(res_to_store)
    # now res is a list; its i-th element contains the value of each parameter (order: rc2, rc3, qg) and a
    # list with all the frequencies flattened
    
    maxes = [rc2s[-1], rc3s[-1], qgs[-1]]
    res_var = []
    for i in range(len(res[0]) - 1):
        tmp_res_var = []
        for sres in res:
            pars = sres[:3] # parameters used for sres
            pars.pop(i) # remove the varying parameter
            comp = maxes.copy() # maximum paramters
            comp.pop(i) # remove the varying parameter
            if all([x == y for x,y in zip(comp, pars)]): # check that the fixed parameters of sres are the maximum ones
                tmp_res_var.append(sres)
        res_var.append(tmp_res_var)
        # now res_var is a list. the i-th element contains the results with the maximum parameters different
        # from the i-th, while the i-th parameter varies each time over all its possible values.
        # Parameters are in the following order: rc2, rc3, qg

    errs = []
    for i in range(len(res_var)):
        tmp_err = []
        for j in range(1, len(res_var[i])):
            tmp_err.append([res_var[i][j][i], comp_err(res_var[i][j-1][-1], res_var[i][j][-1])])
        errs.append(tmp_err)

    x = [np.array([x[0] for x in y]) for y in errs]
    y = [np.array([np.absolute(x[-1]).max() for x in y]) for y in errs]
    return x, y



def parse_dielectric_tensors(filepath=Path('run.abo')):
    '''
    Function to parse the dielectric tensor from the abinit abo file
    '''
    #print(filepath)
    if filepath is not None:
        filepath = Path(filepath)
    with open(filepath.absolute(), 'r') as fl:
        lines = fl.readlines()
    for i, line in enumerate(lines):
        if 'Dielectric tensor' in line:
            extract = lines[i + 4: i + 4 + 10 + 1]
            extract.pop(3)
            extract.pop(6)
    mat = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            mat[i,j] = float(extract[i*3 + j].split()[4])
    return mat


def write_dielectric_tensors(natoms, wdir=Path('./'), atomic=False, outfilepath=None):
    '''
    Function to write the input file for the raman calculation 'infile.dielectric_tensor'.
    There are two possibilities: atomic displacements and mode displacements.
    Atomic displacements: the function looks for 3N folders with the name 'displacement_XXX_Y/ZZZ' in the working directory,
                          where XXX is the number of the atom (starting from 1, max 1000 atoms), Y is the cartesian direction
                          of displacement (x, y and z), and ZZZ is either 'plus' or 'minus'. Inside each of such directories,
                          the function looks for a 'run.abo' file and checks if there is the 'Overall time' string inside. If 
                          everything is ok, then the function extracts all the dielectric tensors (only real parts) writing in
                          each line a row of the tensor, stacking plus and minus tensors, stacking the pair of tensors (p/m)
                          for all the atomic displacements.
    Mode displacements: the function looks for 3N - 3 folders with the name 'X_mode/ZZZ' in the workin directory, where X is 
                        the index of the optical mode (starting from X=3), and ZZZ is either 'plus' or 'minus'.  Inside each 
                        of such directories, the function looks for a 'run.abo' file and checks if there is the 'Overall time'
                        string inside. If everything is ok, then the function extracts all the dielectric tensors (only real
                        parts) writing in each line a row of the tensor, stacking plus and minus tensors, stacking the pair of
                        tensors (p/m) for all the mode displacements.
    '''
    
    if wdir is not None:
        wdir = Path(wdir)
        
    if outfilepath is None:
        outfilepath = wdir.joinpath('infile.dielectric_tensor')
        
    if atomic == False: # MODE DISPLACEMENTS
        # check that all the modes are there
        bad = 0
        for i in range(3, natoms * 3):
            mode_dir = wdir.joinpath(f'{i}_mode')
            if not mode_dir.exists():
                print(f"The folder '{mode_dir.absolute()}' is missing.")
                bad = 1
            else:
                for pm in ['plus', 'minus']:
                    displ_dir = mode_dir.joinpath(f'{pm}/run.abo')
                    if not displ_dir.exists():
                        print(f'There is no .abo file for the {pm} displacement of mode {i} ({displ_dir.absolute()})')
                        bad = 1
                    else:
                        with open(displ_dir.absolute(), 'r') as fl:
                            if all(['Overall time' not in x for x in fl.readlines()]):
                                print(f'The calculation of the {pm} displacement of the mode {i} is not completed!')
                        bad = 1
        if bad == 1:
            exit()
        
        txt = ''
        for i in range(3, natoms*3):
            # positive mode displ.
            filepath = wdir.joinpath(f'{i}_mode/plus/run.abo')
            tens_p = parse_dielectric_tensors(filepath.absolute())
            
            # negative mode displ.
            filepath = wdir.joinpath(f'{i}_mode/minus/run.abo')
            tens_m = parse_dielectric_tensors(filepath.absolute())
            
            # write both
            txt += f'{tens_p[0,0]:.5f} {tens_p[0,1]:.5f} {tens_p[0,2]:.5f}\n'
            txt += f'{tens_p[1,0]:.5f} {tens_p[1,1]:.5f} {tens_p[1,2]:.5f}\n'
            txt += f'{tens_p[2,0]:.5f} {tens_p[2,1]:.5f} {tens_p[2,2]:.5f}\n'
            
            txt += f'{tens_m[0,0]:.5f} {tens_m[0,1]:.5f} {tens_m[0,2]:.5f}\n'
            txt += f'{tens_m[1,0]:.5f} {tens_m[1,1]:.5f} {tens_m[1,2]:.5f}\n'
            txt += f'{tens_m[2,0]:.5f} {tens_m[2,1]:.5f} {tens_m[2,2]:.5f}\n'
        with open(outfilepath.absolute(), 'w') as fl:
            fl.write(txt)
            
            
    else: # ATOMIC DISPLACEMENTS
        
        # check that all the atomic displacements are there
        bad = 0
        for i in range(1, natoms+1):
            for d in ['x', 'y', 'z']:
                displ_dir = wdir.joinpath(f'displacement_{i:0{3}d}_{d}')
                if not displ_dir.exists():
                    print(f"The folder '{displ_dir.absolute()}' is missing.")
                    bad = 1
                else:
                    for pm in ['plus', 'minus']:
                        filepath = displ_dir.joinpath(f'{pm}/run.abo')
                        if not filepath.exists():
                            print(f'There is no .abo file for the {pm} displacement of atom {i} along {d} ({filepath.absolute()})')
                            bad = 1
                        else:
                            with open(filepath.absolute()) as fl:
                                if all(['Overall time' not in x for x in fl.readlines()]):
                                    print(f'The calculation of the {pm} displacement of the mode {i} along {d} is not completed!')
                                    bad = 1
        if bad == 1:
            #exit()
            pass
        
        txt = ''
        for i in range(1, natoms+1):
            for d in ['x', 'y', 'z']:
                # positive atomic displ.
                filepath = wdir.joinpath(f'displacement_{i:0{3}d}_{d}/plus/run.abo')
                tens_p = parse_dielectric_tensors(filepath.absolute())

                # negative atomic displ.
                filepath = wdir.joinpath(f'displacement_{i:0{3}d}_{d}/minus/run.abo')
                tens_m = parse_dielectric_tensors(filepath.absolute())

                # write both
                txt += f'{tens_p[0,0]:.5f} {tens_p[0,1]:.5f} {tens_p[0,2]:.5f}\n'
                txt += f'{tens_p[1,0]:.5f} {tens_p[1,1]:.5f} {tens_p[1,2]:.5f}\n'
                txt += f'{tens_p[2,0]:.5f} {tens_p[2,1]:.5f} {tens_p[2,2]:.5f}\n'

                txt += f'{tens_m[0,0]:.5f} {tens_m[0,1]:.5f} {tens_m[0,2]:.5f}\n'
                txt += f'{tens_m[1,0]:.5f} {tens_m[1,1]:.5f} {tens_m[1,2]:.5f}\n'
                txt += f'{tens_m[2,0]:.5f} {tens_m[2,1]:.5f} {tens_m[2,2]:.5f}\n'
        with open(outfilepath.absolute(), 'w') as fl:
            fl.write(txt)

            
def new_write_dielectric_tensors(natoms, wdir=Path('./'), atomic=False, outfilepath=None): # ONLY FOR LATEST COMMITS OF TOOLS.TDEP
    '''
    Function to write the input file for the raman calculation 'infile.dielectric_tensor'.
    There are two possibilities: atomic displacements and mode displacements.
    Atomic displacements: the function looks for 3N folders with the name 'displacement_XXX_Y/ZZZ' in the working directory,
                          where XXX is the number of the atom (starting from 1, max 1000 atoms), Y is the cartesian direction
                          of displacement (x, y and z), and ZZZ is either 'plus' or 'minus'. Inside each of such directories,
                          the function looks for a 'run.abo' file and checks if there is the 'Overall time' string inside. If 
                          everything is ok, then the function extracts all the dielectric tensors (only real parts) writing in
                          each line a row of the tensor, stacking plus and minus tensors, stacking the pair of tensors (p/m)
                          for all the atomic displacements.
    Mode displacements: the function looks for 3N - 3 folders with the name 'X_mode/ZZZ' in the workin directory, where X is 
                        the index of the optical mode (starting from X=3), and ZZZ is either 'plus' or 'minus'.  Inside each 
                        of such directories, the function looks for a 'run.abo' file and checks if there is the 'Overall time'
                        string inside. If everything is ok, then the function extracts all the dielectric tensors (only real
                        parts) writing in each line a row of the tensor, stacking plus and minus tensors, stacking the pair of
                        tensors (p/m) for all the mode displacements.
    '''
    
    if wdir is not None:
        wdir = Path(wdir)
        
    if outfilepath is None:
        outfilepath = wdir.joinpath('infile.dielectric_tensor')
        
    if atomic == False: # MODE DISPLACEMENTS
        # check that all the modes are there
        bad = False
        for i in range(3, natoms * 3):
            mode_dir = wdir.joinpath(f'{i}_mode')
            if not mode_dir.exists():
                print(f"The folder '{mode_dir.absolute()}' is missing.")
                bad = True
            else:
                for pm in ['plus', 'minus']:
                    displ_dir = mode_dir.joinpath(f'{pm}/run.abo')
                    if not displ_dir.exists():
                        print(f'There is no .abo file for the {pm} displacement of mode {i} ({displ_dir.absolute()})')
                        bad = True
                    else:
                        with open(displ_dir.absolute(), 'r') as fl:
                            if all(['Overall time' not in x for x in fl.readlines()]):
                                print(f'The calculation of the {pm} displacement of the mode {i} is not completed!')
                        bad = True
        if bad == True:
            exit()
        
        txt = ''
        for i in range(3, natoms*3):
            # positive mode displ.
            filepath = wdir.joinpath(f'{i}_mode/plus/run.abo')
            tens_p = parse_dielectric_tensors(filepath.absolute())
            
            # negative mode displ.
            filepath = wdir.joinpath(f'{i}_mode/minus/run.abo')
            tens_m = parse_dielectric_tensors(filepath.absolute())
            
            # write both
            txt += f'{tens_p[0,0]:.5f} {tens_p[0,1]:.5f} {tens_p[0,2]:.5f}\n'
            txt += f'{tens_p[1,0]:.5f} {tens_p[1,1]:.5f} {tens_p[1,2]:.5f}\n'
            txt += f'{tens_p[2,0]:.5f} {tens_p[2,1]:.5f} {tens_p[2,2]:.5f}\n'
            
            txt += f'{tens_m[0,0]:.5f} {tens_m[0,1]:.5f} {tens_m[0,2]:.5f}\n'
            txt += f'{tens_m[1,0]:.5f} {tens_m[1,1]:.5f} {tens_m[1,2]:.5f}\n'
            txt += f'{tens_m[2,0]:.5f} {tens_m[2,1]:.5f} {tens_m[2,2]:.5f}\n'
        with open(outfilepath.absolute(), 'w') as fl:
            fl.write(txt)
            
            
    else: # ATOMIC DISPLACEMENTS
        
        # check that all the atomic displacements are there
        js = ['x', 'y', 'z']
        ks = [None, 'plus', 'minus'] # the first one is dummy, we just need to start from 1
        bad = False
        for i in range(1, natoms+1):
            for j in range(3):
                for k in [1,2]:
                    num = natoms*(i-1) + 2*j + k

                    displ_dir = wdir.joinpath(f'displacement_{num:0{5}d}_{js[j]}_{ks[k]}')
                    if not displ_dir.exists():
                        print(f"The folder '{displ_dir.absolute()}' is missing.")
                        bad = True

                    filepath = displ_dir.joinpath('run.abo')
                    if not filepath.exists():
                        print(f'There is no .abo file for the {ks[k]}-th displacement of atom {i} along {js[j]} ({filepath.absolute()})')
                        bad = True
                    else:
                        with open(filepath.absolute()) as fl:
                            if all(['Overall time' not in x for x in fl.readlines()]):
                                print(f'The calculation of the {ks[k]}-th displacement of atom {i} along {js[j]} ({filepath.absolute()}) is not completed!')
                                bad = True
        if bad == True:
            exit()
            #pass
        
        txt = ''
        for i in range(1, natoms+1):
            for j in range(3):

                # positive atomic displ.
                k = 1
                num = natoms*(i-1) + 2*j + k
                filepath = wdir.joinpath(f'displacement_{num:0{5}d}_{js[j]}_plus/run.abo')
                tens_p = parse_dielectric_tensors(filepath.absolute())

                # negative atomic displ.
                k = 2
                num += 1
                filepath = wdir.joinpath(f'displacement_{num:0{5}d}_{js[j]}_minus/run.abo')
                tens_m = parse_dielectric_tensors(filepath.absolute())

                # write both
                txt += f'{tens_p[0,0]:.5f} {tens_p[0,1]:.5f} {tens_p[0,2]:.5f}\n'
                txt += f'{tens_p[1,0]:.5f} {tens_p[1,1]:.5f} {tens_p[1,2]:.5f}\n'
                txt += f'{tens_p[2,0]:.5f} {tens_p[2,1]:.5f} {tens_p[2,2]:.5f}\n'

                txt += f'{tens_m[0,0]:.5f} {tens_m[0,1]:.5f} {tens_m[0,2]:.5f}\n'
                txt += f'{tens_m[1,0]:.5f} {tens_m[1,1]:.5f} {tens_m[1,2]:.5f}\n'
                txt += f'{tens_m[2,0]:.5f} {tens_m[2,1]:.5f} {tens_m[2,2]:.5f}\n'
        with open(outfilepath.absolute(), 'w') as fl:
            fl.write(txt)

            
def extract_diel_bec(filepath):
    '''
    Function to extract the dielectric tensor and Born effective charges from an .abo of an abinit calculation
    (NOT from anaddb!)
    Args:
    filepath(str): path to the .abo
    Return:
    dielt: numpy array (2x2) with the dielectric tensor
    bec: list (length=natom) of numpy arrays (2x2) containing the bec for each atom; bec[i][j,k] = bec of atom i
         with electric field along j and displacement along k
    '''

    diel_token = 0
    bec_token = 0
    natoms_token = 0
    
    filepath = Path(filepath)
    with open(filepath.absolute(), 'r') as fl:
        lines = fl.readlines()
    for i, line in enumerate(lines):
        if bec_token == 1 and natoms_token == 1 and diel_token == 1:
            break

        if 'natom:' in line and diel_token == 0:
            natoms = int(line.split()[2][:-1])
            natoms_token = 1

        if 'Dielectric tensor' in line and diel_token == 0:
            dielt = np.zeros((3,3))
            ps = [4, 5, 6, 8, 9, 10, 12, 13, 14]
            for p in ps:
                j = int(lines[i+p].split()[0]) - 1
                k = int(lines[i+p].split()[2]) - 1
                dielt[j,k] += float(lines[i+p].split()[4])
            diel_token = 1

        elif 'Effective charges' in line and bec_token == 0 and 'from phonon' in lines[i+1]:
            bec = [np.zeros((3,3)) for _ in range(natoms)]
            s = 0
            for j1 in range(3*natoms):
                for j2 in range(3):
                    m = int(lines[i + 6 + j1*4 + j2].split()[3]) - 1 # atom
                    j = int(lines[i + 6 + j1*4 + j2].split()[0]) - 1 # elf dir
                    k = int(lines[i + 6 + j1*4 + j2].split()[2]) - 1 # displ dir
                    bec[m][j,k] += float(lines[i + 6 + j1*4 + j2].split()[4]) # real part of the bec   
            bec_token = 1
    return dielt, bec


def write_lotofile(inpath, outpath='./infile.lotosplitting'):
    '''
    Function to extract the dielectric tensor and Born effective charges from an .abo of an abinit calculation
    (NOT from anaddb!) and write them into a "infile.lotosplitting" (the name can be changed) for tdep.
    Args:
    inpath(str): path to the .abo
    outpath(str): path (with name) of the outputfile
    '''
    inpath = Path(inpath)
    if outpath is not None:
        outpath = Path(outpath)
    
    dielt, becs = extract_diel_bec(inpath.absolute())
    txt = ''
    for i in range(3):
            txt += f'{dielt[i,0]:.10f} {dielt[i,1]:.10f} {dielt[i,2]:.10f}\n'
    for bec in becs:
        for i in range(3):
            txt += f'{bec[i,0]:.10f} {bec[i,1]:.10f} {bec[i,2]:.10f}\n'
    with open(outpath.absolute(), 'w') as fl:
        fl.write(txt)

def convergence_tdep_stride(stride=True,
                            sampling_size=False,
                            temperature=0,
                            root_dir='./',
                            tdep_bin_directory = Path(g_tdep_bin_directory),
                            bin_prefix = 1,
                            first_order = True,
                            displ_threshold_firstorder = 0.0001,
                            max_iterations_first_order = 20,
                            nthrow = 0,
                            rc2 = 10,
                            rc3 = 5,
                            ts = 1,
                            max_stride=None,
                            stride_step=None,
                            uc_path = 'unitcell.json',
                            mult_mat = [[1,0,0],[0,1,0],[0,0,1]],
                            traj_path = './Trajectory.traj',
                            polar = False,
                            loto_filepath = None,
                            job=False,
                            job_template=None):

    # take all the parameters at once
    parameters = locals().copy # dictionary

    parameters['stride'] = True
    parameters['sampling_size'] = False

    convergence_tdep_stride_or_sampling_size(**parameters)

def convergence_tdep_sampling_size(stride=True,
                                   sampling_size=False,
                                   temperature=0,
                                   root_dir='./',
                                   tdep_bin_directory = Path(g_tdep_bin_directory),
                                   bin_prefix = 1,
                                   first_order = True,
                                   displ_threshold_firstorder = 0.0001,
                                   max_iterations_first_order = 20,
                                   nthrow = 0,
                                   rc2 = 10,
                                   rc3 = 5,
                                   ts = 1,
                                   size_step=None,
                                   stride_for_size_conv=1,
                                   uc_path = 'unitcell.json',
                                   mult_mat = [[1,0,0],[0,1,0],[0,0,1]],
                                   traj_path = './Trajectory.traj',
                                   polar = False,
                                   loto_filepath = None,
                                   job=False,
                                   job_template=None):

    # take all the parameters at once
    parameters = locals().copy # dictionary

    parameters['stride'] = False
    parameters['sampling_size'] = True

    convergence_tdep_stride_or_sampling_size(**parameters)

def convergence_tdep_stride_and_sampling_size(temperature=0,
                                              root_dir='./',
                                              tdep_bin_directory = Path(g_tdep_bin_directory),
                                              bin_prefix = 1,
                                              first_order = True,
                                              displ_threshold_firstorder = 0.0001,
                                              max_iterations_first_order = 20,
                                              nthrow = 0,
                                              rc2 = 10,
                                              rc3 = 5,
                                              ts = 1,
                                              size_step=None,
                                              stride_for_size_conv=1,
                                              max_stride=None,
                                              stride_step=None,
                                              uc_path = 'unitcell.json',
                                              mult_mat = [[1,0,0],[0,1,0],[0,0,1]],
                                              traj_path = './Trajectory.traj',
                                              polar = False,
                                              loto_filepath = None,
                                              job=False,
                                              job_template=None):
    
    # take all the parameters at once
    parameters = locals().copy # dictionary

    parameters['stride'] = True
    parameters['sampling_size'] = True

    convergence_tdep_stride_or_sampling_size(**parameters)
    

def convergence_tdep_stride_or_sampling_size(stride=True,
                                             sampling_size=True,
                                             temperature=0,
                                             root_dir='./',
                                             tdep_bin_directory = Path(g_tdep_bin_directory),
                                             bin_prefix = 1,
                                             first_order = True,
                                             displ_threshold_firstorder = 0.0001,
                                             max_iterations_first_order = 20,
                                             nthrow = 0,
                                             rc2 = 10,
                                             rc3 = 5,
                                             ts = 1,
                                             size_step=None,
                                             stride_for_size_conv=1,
                                             max_stride=None,
                                             stride_step=None,
                                             uc_path = 'unitcell.json',
                                             mult_mat = [[1,0,0],[0,1,0],[0,0,1]],
                                             traj_path = './Trajectory.traj',
                                             traj_size = 'full',
                                             polar = False,
                                             loto_filepath = None,
                                             job=False,
                                             job_template=None):
    
    
    template = Path(__file__).parent.joinpath('data/conv_tdep/Run_tdep_template.py')
    with open(template, 'r') as fl:
        lines = fl.readlines()

    index_wdir = None
    index_index = None
    
    for i in range(28):
        if '$wdir$' in lines[i]:
            index_wdir = i # just to note where the wdir line is in the file
        elif '$tdep_bin_directory$' in lines[i]:
            lines[i] = f"tdep_bin_directory = '{tdep_bin_directory.absolute()}'"
        elif '$bin_prefix$' in lines[i]:
            lines[i] = f'bin_prefix = \'{bin_prefix}\''
        elif '$nthrow$' in lines[i]:
            lines[i] = f'nthrow = {nthrow}'
        elif '$rc2$' in lines[i]:
            lines[i] = f'rc2 = {rc2}'
        elif '$rc3$' in lines[i]:
            lines[i] = f'rc3 = {rc3}'
        elif '$ts$' in lines[i]:
            lines[i] = f'ts = {ts}'
        elif '$traj_path$' in lines[i]:
            lines[i] = f"traj_path = '{traj_path.absolute()}'"
        elif '$uc_path$' in lines[i]:
            lines[i] = f"uc_path = '{uc_path.absolute()}'"
        elif '$mult_mat$' in lines[i]:
            lines[i] = f'mult_mat = [[{mult_mat[0][0]}, {mult_mat[0][1]}, {mult_mat[0][2]}], '
            lines[i] += f'[{mult_mat[1][0]}, {mult_mat[1][1]}, {mult_mat[1][2]}], '
            lines[i] += f'[{mult_mat[2][0]}, {mult_mat[2][1]}, {mult_mat[2][2]}]]'
        elif '$temperature$' in lines[i]:
            lines[i] = f'temperature = {temperature}'
        elif '$index$' in lines[i]:
            index_index = i # just to note where the index line is in the file
        elif '$polar$' in lines[i]:
            lines[i] = f'polar = {polar}'
        elif '$loto_filepath$' in lines[i]:
            lines[i] = f'loto_filepath = "{Path(loto_filepath.absolute())}"'
        elif '$first_order$' in lines[i]:
            lines[i] = f'first_order = {first_order}'
        elif '$displ_threshold_firstorder$' in lines[i]:
            lines[i] = f'displ_threshold_firstorder = {displ_threshold_firstorder}'
        elif '$max_iterations_first_order$' in lines[i]:
            lines[i] = f'max_iterations_first_order = {max_iterations_first_order}'
        elif '$stride$' in lines[i]:
            index_stride = i # just to note where the stride line is in the file
        else:
            continue
        lines[i] += '\n'

    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    conv_dir = root_dir.joinpath('Convergence_tdep')
    conv_dir.mkdir(parents=True, exist_ok=True)
    ats = read(traj_path, index=':')
    if traj_size != 'full':
        ats = ats[:traj_size]
    nconfs = len(ats)

    if sampling_size == True:
        # deal with the sizes
        if size_step is None:
            raise ValueError('sampling_size is True, but you did not provide size_step!')
        if (nconfs-nthrow) < size_step:
            raise ValueError('size_step must be < than the number of confs in the trajectory file!')
        if stride_for_size_conv is None:
            raise ValueError('sampling_size is True, but you did not provide stride_for_size_conv!')
        sizes_dir = conv_dir.joinpath('sampling_size')
        sizes_dir.mkdir(parents=True, exist_ok=True)
        # find sizes (aka indices) to converge w.r.t.
        indices = [size_step*i for i in range(1, ceil((nconfs-nthrow)/size_step)+1)]
        # apply stride to indices
        indices = [ceil(size/stride_for_size_conv) for size in indices]
        for index in indices:
            inst_dir = sizes_dir.joinpath(f'{index}_size')
            inst_dir.mkdir(parents=True, exist_ok=True)
            lines[index_index] = f"index = {ceil(index/stride_for_size_conv)}\n"
            lines[index_wdir] = f"wdir = '{inst_dir.absolute()}'\n"
            lines[index_stride] = f"stride = {stride_for_size_conv}\n"
            with open(inst_dir.joinpath('RunTdep.py'), 'w') as fl:
                fl.writelines(lines)
            if job == True:
                shutil.copy(Path(job_template).absolute(), inst_dir)
                run(f'sbatch {Path(job_template).name}', cwd=inst_dir, shell=True)

    if stride == True:
        if max_stride is None:
            raise ValueError('stride is True, but you did not provide max_stride!')
        if stride_step is None:
            raise ValueError('stride is True, but you did not provide stride_step!')

        # deal with the strides

        strides_dir = conv_dir.joinpath('strides')
        strides_dir.mkdir(parents=True, exist_ok=True)
        strides = [x for x in range(1, max_stride+1, stride_step)]
        max_stride = max(strides) # actual max stride after rounding 
        stridden_nconfs = ceil((nconfs-nthrow) / max_stride) # the number of confs after applying the stride for all stride values 
        for stride in strides:
            #nconfs_for_stride = stridden_nconfs * stride
            inst_dir = strides_dir.joinpath(f'{stride}_stride')
            inst_dir.mkdir(parents=True, exist_ok=True)
            lines[index_index] = f"index = {stridden_nconfs}\n"
            lines[index_wdir] = f"wdir = '{inst_dir.absolute()}'\n"
            lines[index_stride] = f"stride = {stride}\n"
            with open(inst_dir.joinpath('RunTdep.py'), 'w') as fl:
                fl.writelines(lines)
            if job == True:
                shutil.copy(Path(job_template).absolute(), inst_dir)
                run(f'sbatch {Path(job_template).name}', cwd=inst_dir, shell=True)




def new_convergence_stride(temperature=0,
                           root_dir='./',
                           tdep_bin_directory = Path(g_tdep_bin_directory),
                           bin_prefix = '',
                           first_order = True,
                           displ_threshold_firstorder = 0.0001,
                           max_iterations_first_order = 20,
                           nthrow = 0,
                           rc2 = 10,
                           rc3 = 5,
                           ts = 1,
                           size_step=1000,
                           max_stride=None,
                           stride_step=None,
                           ifc_threshold=0.0001,
                           uc_path = 'unitcell.json',
                           mult_mat = [[1,0,0],[0,1,0],[0,0,1]],
                           traj_path = './Trajectory.traj',
                           polar = False,
                           loto_filepath = None):
    root_dir = Path(root_dir)
    unitcell = read(uc_path)
    supercell = make_supercell(unitcell, mult_mat)
    strides = list(range(1, max_stride, stride_step))
    traj = read(traj_path, index=':')
    sizes = list(range(size_step, len(traj), size_step))
    for stride in strides:
        stridden_traj = traj[::stride]
        shuffle(stridden_traj)
        stride_dir = root_dir.joinpath(f'stride_{stride}')
        stride_dir.mkdir(parents=True, exist_ok=True)
        print(f'****** Stride = {stride} ******')
        print(f'Maximum size with this stride: {len(stridden_traj)} configurations')
        ifcs = []
        for i_s, size in enumerate(sizes):
            if size > len(stridden_traj):
                print(f'Only {len(stridden_traj)} configurations; not enough to try size = {size}')
                print(f'xxxxxx Stride = {stride} did not converge! xxxxxx')
                break
            print(f'------ Stride = {stride}, size = {size} ------')
            size_dir = stride_dir.joinpath(f'size_{size}')
            size_dir.mkdir(parents=True, exist_ok=True)
            extract_ifcs(from_infiles = False,
                        infiles_dir = None,
                        unitcell = unitcell,
                        supercell = supercell,
                        sampling = stridden_traj[:size],
                        timestep = 1,
                        dir = size_dir,
                        first_order = first_order,
                        displ_threshold_firstorder = displ_threshold_firstorder,
                        max_iterations_first_order = max_iterations_first_order,
                        rc2 = rc2, 
                        rc3 = rc3, 
                        polar = polar,
                        loto_filepath = loto_filepath, 
                        stride = stride, 
                        temperature = temperature,
                        bin_prefix = bin_prefix,
                        tdep_bin_directory = tdep_bin_directory)
            ifcs.append(parse_outfile_forceconstants(size_dir.joinpath('outfile.forceconstant'), unitcell=unitcell, supercell=supercell))
            if i_s > 0:
                weights = np.abs(ifcs[-1]) + np.abs(ifcs[-2])
                diffs = np.abs(ifcs[-1] - ifcs[-2])
                avg_diff = (diffs * weights / weights.sum()).sum()
                if avg_diff < ifc_threshold:
                    print(f'Average difference in IFCs (eV/Ang^2) between size {size} and {sizes[i_s-1]} = {avg_diff:.10f} < {ifc_threshold}!')
                    print(f'Convergence reached at size {size}')
                    break
                else:
                    print(f'Average difference in IFCs (eV/Ang^2) between size {size} and {sizes[i_s-1]} = {avg_diff:.10f} >= {ifc_threshold}!')

        





def old_convergence_tdep_mdlen(
        temperature,
        root_dir='./',
        folderbin = Path(g_tdep_bin_directory),
        nproc = 1,
        refine_cell = True,
        nthrow = 0,
        rc2 = 10,
        rc3 = 5,
        U0 = True,
        qg = 32,
        traj_path = './Trajectory.traj',
        size_step=10,
        max_stride=100,
        stride_step=8,
        uc_path = 'unitcell.json',
        mult_mat = [[1,0,0],[0,1,0],[0,0,1]],
        job=False,
        job_template=None):
    
    '''Run Tdep with increasing sampling size and stride to assess convergence.
    The two parameters are converged separately: stride is converged with the maximum size and the size is converged with the minimum stride'''
    
    template = Path(__file__).parent.joinpath('data/conv_tdep/Run_tdep_template.py')
    with open(template, 'r') as fl:
        lines = fl.readlines()
    
    index_wdir = None
    index_index = None
    
    for i in range(27):
        if 'wdir' in lines[i]:
            index_wdir = i
        elif 'folderbin' in lines[i]:
            lines[i] = f"folderbin = '{folderbin.absolute()}'"
        elif 'nproc' in lines[i]:
            lines[i] = f'nproc = {nproc}'
        elif 'refine_cell' in lines[i]:
            lines[i] = f'refine_cell = {refine_cell}'
        elif 'nthrow' in lines[i]:
            lines[i] = f'nthrow = {nthrow}'
        elif 'rc2' in lines[i]:
            lines[i] = f'rc2 = {rc2}'
        elif 'rc3' in lines[i]:
            lines[i] = f'rc3 = {rc3}'
        elif 'U0' in lines[i]:
            lines[i] = f'U0 = {U0}'
        elif 'qg' in lines[i]:
            lines[i] = f'qg = {qg}'
        elif 'traj_path' in lines[i]:
            lines[i] = f"traj_path = '{traj_path.absolute()}'"
        elif 'uc_path' in lines[i]:
            lines[i] = f"uc_path = '{uc_path.absolute()}'"
        elif 'mult_mat' in lines[i]:
            lines[i] = f'mult_mat = [[{mult_mat[0][0]}, {mult_mat[0][1]}, {mult_mat[0][2]}], '
            lines[i] += f'[{mult_mat[1][0]}, {mult_mat[1][1]}, {mult_mat[1][2]}], '
            lines[i] += f'[{mult_mat[2][0]}, {mult_mat[2][1]}, {mult_mat[2][2]}]]'
        elif 'temperature' in lines[i]:
            lines[i] = f'temperature = {temperature}'
        elif 'index' in lines[i]:
            index_index = i
        elif 'nslice' in lines[i]:
            index_slice = i
        else:
            continue
        lines[i] += '\n'

        
    


    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    conv_dir = root_dir.joinpath('Convergence_tdep')
    conv_dir.mkdir(parents=True, exist_ok=True)
    
    ats = read(traj_path, index=':')
    nconfs = len(ats)
    if nconfs < size_step:
        raise ValueError('nstep must be < than the number of confs in the trajectory file!')
    indices = [size_step*i for i in range(1, ceil(nconfs/size_step)+1)]

    
    # deal with the sizes and minimum stride (=1)
    sizes_dir = conv_dir.joinpath('sampling_size')
    sizes_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        inst_dir = sizes_dir.joinpath(f'{index}_size')
        inst_dir.mkdir(parents=True, exist_ok=True)
        lines[index_index] = f"index = {index}\n"
        lines[index_wdir] = f"wdir = '{inst_dir.absolute()}'\n"
        lines[index_slice] = f"nslice = 1\n"
        with open(inst_dir.joinpath('RunTdep.py'), 'w') as fl:
            fl.writelines(lines)
        if job == True:
            shutil.copy(Path(job_template).absolute(), inst_dir)
            run(f'sbatch {Path(job_template).name}', cwd=inst_dir, shell=True)

    # deal with the strides and maximum size
    strides_dir = conv_dir.joinpath('strides')
    strides_dir.mkdir(parents=True, exist_ok=True)
    strides = [x for x in range(1, max_stride, stride_step)]
    for stride in strides:
        inst_dir = strides_dir.joinpath(f'{stride}_stride')
        inst_dir.mkdir(parents=True, exist_ok=True)
        lines[index_index] = f"index = {indices[-1]}\n"
        lines[index_wdir] = f"wdir = '{inst_dir.absolute()}'\n"
        lines[index_slice] = f"nslice = 1\n"
        with open(inst_dir.joinpath('RunTdep.py'), 'w') as fl:
            fl.writelines(lines)
        if job == True:
            shutil.copy(Path(job_template).absolute(), inst_dir)
            run(f'sbatch {Path(job_template).name}', cwd=inst_dir, shell=True)


def extract_ifcs(from_infiles = False,
                 infiles_dir = None,
                 unitcell = None,
                 supercell = None,
                 sampling = None,
                 timestep = 1,
                 dir = './',
                 first_order = False,
                 displ_threshold_firstorder = 0.0001,
                 max_iterations_first_order = 20,
                 rc2 = None, 
                 rc3 = None, 
                 polar = False,
                 loto_filepath = None, 
                 stride = 1, 
                 temperature = None,
                 bin_prefix = '',
                 tdep_bin_directory = None):
    """
    Extract second- and optionally third-order interatomic force constants (IFCs) using TDEP.

    This function is used to launch the extraction of the IFCs with TDEP input. Optionally, a first-order optimization of the unit cell can be 
    performed prior to the extraction.

    Parameters
    ----------
    from_infiles : bool
        If True, read input files from `infiles_dir`. Otherwise, generate them from provided data.
    infiles_dir : str or Path
        Directory containing the TDEP input files (required if `from_infiles` is True).
    unitcell : ase.Atoms
        Unit cell structure (ignored if `from_infiles` is True).
    supercell : ase.Atoms
        Supercell structure (ignored if `from_infiles` is True).
    sampling : list of ase.Atoms
        List of trajectory frames used to generate positions and forces.
    timestep : float
        Time step of the MD sampling (in femtoseconds).
    dir : str or Path
        Working directory where the IFCs will be extracted.
    first_order : bool
        If True, perform a first-order TDEP optimization of the unit cell before extracting IFCs.
    displ_threshold_firstorder : float
        Maximum allowed displacement threshold in the first-order optimization (in Å).
    max_iterations_first_order : int
        Maximum number of iterations for first-order optimization.
    rc2 : float
        Second-order cutoff radius (in Å). Required.
    rc3 : float or None
        Third-order cutoff radius (in Å). If None, only second-order IFCs are computed.
    polar : bool
        Whether to include LO-TO splitting corrections.
    loto_filepath : str or Path
        Path to the file with LO-TO splitting info (required if `polar` is True).
    stride : int
        Stride used to sample the MD trajectory.
    temperature : float
        Temperature at which to extract IFCs (in K).
    bin_prefix : str
        Optional prefix to prepend to TDEP binary calls (e.g., for MPI).
    tdep_bin_directory : str or Path
        Path to the TDEP binaries.

    Raises
    ------
    TypeError
        If mandatory parameters are missing or inconsistent.
    """
    
    
    # FUNCTIONS #
    def check_convergence_positions(dir, displ_threshold_firstorder):
        old_positions = read(dir.joinpath('infile.ucposcar'), format='vasp').get_positions()
        new_positions = read(dir.joinpath('outfile.new_ucposcar'), format='vasp').get_positions()
        err = new_positions - old_positions

        if (err >= displ_threshold_firstorder).any():
            return False, err
        else:
            return True, err
        
    def move(src, dest):
        at = read(src, format='vasp')
        write(dest, at, format='vasp')
        src.unlink()
    #############


    print_kb('**** Extraction of IFCs with TDEP ****')


    dir = Path(dir)
    if not dir.is_dir():
        dir.mkdir(parents=True, exist_ok=True)


    if from_infiles == True:
        if infiles_dir is None:
            raise TypeError('Since from_infiles is True, you must provide infiles_dir!')
        else:
            src_infiles_dir = Path(infiles_dir)
    else:
        if any([unitcell is None, supercell is None, sampling is None]):
            raise TypeError('Since from_infiles is False, you must provide the unitcell, the supercell and the sampling trajectory!')
        

    if rc2 is None:
        raise TypeError('rc2 is mandatory!')
    else:
        rc2_cmd = f'-rc2 {rc2}'
    
    if rc3 is None:
        rc3_cmd = ''
    else:
        rc3_cmd = f'-rc3 {rc3}'
    
    if polar == True:
        if loto_filepath is None:
            raise TypeError('Since polar is True, you must provide loto_filepath!')
        else:
            polar_cmd = '--polar'
            loto_filepath = Path(loto_filepath)
    else:
        polar_cmd = ''

    if tdep_bin_directory is None or tdep_bin_directory == '':
        tdep_bin_directory = g_tdep_bin_directory
    else:
        tdep_bin_directory = Path(tdep_bin_directory) # if it's already a Path(), this won't change anything
    
    stride_cmd = f'--stride {stride}'

    temperature_cmd = f'--temperature {temperature}'


    #infiles_dir = Path(dir.joinpath('infiles'))
    #infiles_dir.mkdir(parents=True, exist_ok=True)

    if from_infiles == True:
        ln_s_f(src_infiles_dir.joinpath('infile.ucposcar'), dir)
        os.system(f'ls {dir}')
        ln_s_f(src_infiles_dir.joinpath('infile.ssposcar'), dir)
        ln_s_f(src_infiles_dir.joinpath('infile.meta'), dir)
        ln_s_f(src_infiles_dir.joinpath('infile.stat'), dir)
        ln_s_f(src_infiles_dir.joinpath('infile.positions'), dir)
        ln_s_f(src_infiles_dir.joinpath('infile.forces'), dir)
        unitcell = read(src_infiles_dir.joinpath('infile.ucposcar'), format='vasp')
        supercell = read(src_infiles_dir.joinpath('infile.ssposcar'), format='vasp')
    else:       
        write(dir.joinpath('infile.ucposcar'), unitcell, format='vasp')
        write(dir.joinpath('infile.ssposcar'), supercell, format='vasp')
        make_stat(sampling, dir)
        make_meta(sampling, dir, timestep=timestep, temp=temperature)
        make_forces(sampling, dir)
        make_positions(sampling, dir)

    if polar == True:
        ln_s_f(loto_filepath, dir.joinpath('infile.lotosplitting'))
    
    if first_order == True:
        print_b('You asked for the first-order TDEP optimization of the unitcell; it will be done using the value of rc2 you provided.')
        fo_dir = dir.joinpath('first_order_optimisation')
        unitcell, supercell, converged = first_order_optimization(from_infiles = from_infiles,
                                                    infiles_dir = infiles_dir,
                                                    unitcell = unitcell,
                                                    supercell = supercell,
                                                    sampling = sampling,
                                                    timestep = timestep,
                                                    dir = fo_dir,
                                                    displ_threshold_firstorder = displ_threshold_firstorder,
                                                    max_iterations_first_order = max_iterations_first_order,
                                                    rc2 = rc2,
                                                    polar = polar,
                                                    loto_filepath = loto_filepath,
                                                    stride = stride,
                                                    temperature = temperature,
                                                    bin_prefix = bin_prefix,
                                                    tdep_bin_directory = tdep_bin_directory)

        if converged == False:
            print(colored('The first-order optimization did not converge. However, the last unitcell generated in the process will be used.', 'yellow'))
        ln_s_f(fo_dir.joinpath(f'optimized_unitcell.poscar'), dir.joinpath('infile.ucposcar'))
        ln_s_f(fo_dir.joinpath(f'optimized_supercell.poscar'), dir.joinpath('infile.ssposcar'))

    print('Here are the parameters for the IFCs extraction:')
    print(f'\t2nd-order cutoff: ' +colored(f'{rc2} Angstroms', 'blue'))
    print(f'\tTemperature: ' +colored(f'{temperature}', 'blue'))
    if rc3 == None:
        print('\tNo 3-rd order interactions')
    else:
        print('\t3rd-order cutoff: ' + colored(f'{rc3} Angstroms', 'blue'))
    
    print('\tNo 4th-order interactions') # this could be implemented quite easily; it should laready be in TDEP

    if bin_prefix == '':
        print(f'\tNo binary prefix')
    else:
        print(f'\tBinary prefix: ' +colored(f'{bin_prefix}', 'blue'))
    if polar == True:
        print('\tPolar correction (LO-TO): ' + colored('yes', 'blue'))
    else:
        print('\tPolar correction (LO-TO): ' + colored('no', 'blue'))
    print(f'\tStride: ' +colored(f'{stride}', 'blue'))
    
    cmd = f'{bin_prefix} {tdep_bin_directory.joinpath("extract_forceconstants")} {rc2_cmd} {rc3_cmd} {temperature_cmd} {polar_cmd} {stride_cmd}'

    logpath = dir.joinpath('log_ifc')
    errpath = dir.joinpath('err_ifc')
    with open(logpath.absolute(), 'w') as log, open(errpath.absolute(), 'w') as err:
        run(cmd.split(), cwd=dir.absolute(), stdout=log, stderr=err)
    print_kb('****** ' + colored('Extraction of IFCs done!', 'green') + ' ******') 

            
def conv_rc2_extract_ifcs(unitcell = None,
                          supercell = None,
                          sampling = None,
                          timestep = 1,
                          dir = './',
                          first_order = False,
                          first_order_rc2 = None,
                          displ_threshold_firstorder = 0.0001,
                          max_iterations_first_order = 20,
                          rc2s = None, 
                          #rc3 = None, 
                          polar = False,
                          loto_filepath = None, 
                          stride = 1, 
                          temperature = None,
                          bin_prefix = '',
                          tdep_bin_directory = None,
                          ifc_diff_threshold = 0.001, # eV/A^2
                          n_rc2_to_average = 4,
                          conv_criterion_diff = 'avg'): 
    
    """Function to extract the IFCs converging them w.r.t. the rc2 values.
    It is possible to run a first-order TDEP optimisation of the unitcell before extracting the IFCs and it will be done using a rc2
    specific for this stage (`first_order_rc2` or `max(rc2s)` if the first one is None). The same optimised unitcell is then used for all
    the IFCs extractions with the several rc2 (in `rc2s`).
    The convergence of the IFCs is assessed by evaluating the average differences in the IFCs between the last rc2 value, and the N before,
    and averaging these N average differences (N = `n_rc2_to_average`).  

    Parameters
    ----------
    unitcell : ase.Atoms
        The unit cell.
    supercell : ase.Atoms
        The supercell: must be obtained by repetition of the unitcell via a matrix multiplication!.
    sampling : list or ASE trajectory
        The trajectory used for statistical sampling.
    timestep : float
        Timestep in fs.
    dir : str or Path
        Root directory in which to perform the convergence process.
    first_order : bool
        If True, run a first-order TDEP optimization of the unitcell.
    first_order_rc2 : float or None
        The rc2 cutoff to be used for the first-order optimization.
        If None, the maximum value in `rc2s` will be used.
    displ_threshold_firstorder : float
        Convergence threshold for atomic displacements during first-order optimization (in Ang).
    max_iterations_first_order : int
        Maximum number of iterations allowed in the first-order optimization.
    rc2s : list of float
        List of second-order cutoff radii (in Ang) to test for convergence.
    polar : bool
        Whether to include LO-TO splitting (polar correction).
    loto_filepath : str or Path
        Path to the LO-TO splitting file, required if `polar=True`.
    stride : int
        Stride for sampling frames in TDEP input generation.
    temperature : float
        Temperature (in K).
    bin_prefix : str
        Optional prefix to prepend to TDEP binary calls (e.g. "srun -n 4").
    tdep_bin_directory : str or Path
        Path to the TDEP binaries.
    ifc_diff_threshold : float
        Threshold (in eV/Ang^2) for convergence assessment.
    n_rc2_to_average : int
        Number of previous `rc2` IFCs to average over for convergence assessment.
    conv_criterion_diff : str
        Criterion for convergence: 'avg' for average difference, 'max' for max difference.

    Returns
    -------
    float
        The value of `rc2` (from `rc2s`) at which the IFCs are considered converged.
        If convergence is not reached, returns the last rc2 value tested.

    Raises
    ------
    ValueError
        If input consistency checks fail (e.g., too few rc2s, invalid convergence criterion).
    
    Notes
    -----
    - IFCs are extracted in subfolders named `rc2_<value>` under the main `dir`.
    - A convergence plot (avg/max IFC error vs rc2) is saved as `Convergence.png`.
    - All generated input files are stored in the `infiles` subdirectory.
    - If `first_order=True`, the optimized unitcell is reused for all `rc2` values.
    """
    print_kb('**** IFCs convergence with respect to the 2nd-order cutoff ****')

    dir = Path(dir)
    if not dir.is_dir():
        dir.mkdir(parents=True, exist_ok=True)

    infiles_dir = dir.joinpath('infiles')
    infiles_dir.mkdir(parents=True, exist_ok=True)
    write(infiles_dir.joinpath('infile.ucposcar'), unitcell, format='vasp')
    write(infiles_dir.joinpath('infile.ssposcar'), supercell, format='vasp')
    make_stat(sampling, infiles_dir)
    make_meta(sampling, infiles_dir, timestep=timestep, temp=100)
    make_forces(sampling, infiles_dir)
    make_positions(sampling, infiles_dir)

    if conv_criterion_diff not in ['avg', 'max']:
        raise ValueError('conv_criterion_diff must be either \'avg\' or \'max\'!')
    
    if len(rc2s) <= n_rc2_to_average:
        raise ValueError(f'You asked to assess the convergence by averaging the last {n_rc2_to_average} extractions, but you provided less than {n_rc2_to_average} + 1 values of rc2!')
    
    if first_order == True:
        if first_order_rc2 == None:
            first_order_rc2 = max(rc2s)
            print_b('You asked for the first-order TDEP optimization of the unitcell; it will be done using the biggest value of rc2 you provided.')
        else:
            print_b(f'You asked for the first-order TDEP optimization of the unitcell; it will be done using the rc2 value you provided for this stage ({first_order_rc2} Angstroms).')
        
        fo_dir = dir.joinpath('first_order_optimisation')
        optimised_ucell, optimised_scell, converged = first_order_optimization(from_infiles = False,
                                                   unitcell = unitcell,
                                                   supercell = supercell,
                                                   sampling = sampling,
                                                   timestep = timestep,
                                                   dir = fo_dir,
                                                   displ_threshold_firstorder = displ_threshold_firstorder,
                                                   max_iterations_first_order = max_iterations_first_order,
                                                   rc2 = first_order_rc2,
                                                   polar = polar,
                                                   loto_filepath = loto_filepath,
                                                   stride = stride,
                                                   temperature = temperature,
                                                   bin_prefix = bin_prefix,
                                                   tdep_bin_directory = tdep_bin_directory)
        if converged == False:
            print(colored('The first-order optimization did not converge. However, the last unitcell generated in the process will be used.', 'yellow'))
        unitcell = optimised_ucell
        supercell = optimised_scell
        ln_s_f(fo_dir.joinpath(f'optimized_unitcell.poscar'), infiles_dir.joinpath('infile.ucposcar'))
        ln_s_f(fo_dir.joinpath(f'optimized_supercell.poscar'), infiles_dir.joinpath('infile.ssposcar'))

    print('Starting the extraction for each rc2 value')

    ifcs = []
    for i_rc2, rc2 in enumerate(rc2s):
        print_b(f'+++ rc2 = {rc2} Angstroms +++')
        rc2_dir = dir.joinpath(f'rc2_{rc2}')
        rc2_dir.mkdir(parents=True, exist_ok=True)
        extract_ifcs(from_infiles = True,
                infiles_dir = infiles_dir,
                unitcell = None, # no need
                supercell = None, # no need
                sampling = None, # no need
                timestep = timestep,
                dir = rc2_dir,
                first_order = False,
                displ_threshold_firstorder = None,
                max_iterations_first_order = None,
                rc2 = rc2, 
                #rc3 = rc3, 
                polar = polar,
                loto_filepath = loto_filepath, 
                stride = stride, 
                temperature = temperature,
                bin_prefix = bin_prefix,
                tdep_bin_directory = tdep_bin_directory)
        
        new_ifcs = parse_outfile_forceconstants(rc2_dir.joinpath('outfile.forceconstant'), unitcell, supercell) # shape: n_atoms_ucell, n_atoms_scell, 3, 3
        ifcs.append(new_ifcs)  

        converged = False
        if i_rc2 >= n_rc2_to_average:
            print('Assessing the convergence')
            cifcs = np.array(ifcs.copy()) # "c" stands for copy; shape: n_rc2_done, n_atoms_ucell, n_atoms_scell, 3, 3
            diffss = np.abs(cifcs[-1][np.newaxis,:] - cifcs[-1-n_rc2_to_average:-1]) # shape: n_rc2_to_average, n_atoms_ucell, n_atoms_scell, 3, 3
            weights = np.abs(cifcs[-1][np.newaxis,:]) + np.abs(cifcs[-1-n_rc2_to_average:-1])
            avg_diffss = (diffss * weights / weigths.sum()).sum(axis=(1,2,3,4)) # shape: n_rc2_to_average
            avg_avg_diff = np.mean(avg_diffss)
            max_diffss = np.max(diffss, axis=(1,2,3,4)) # shape: n_rc2_to_average
            avg_max_diff = (np.mean(max_diffss))
            
            if conv_criterion_diff == 'avg':
                value_to_compare_txt = 'average'
                value_to_compare = avg_avg_diff
            elif conv_criterion_diff == 'max':
                value_to_compare_txt = 'maximum'
                value_to_compare = avg_max_diff
            
            print(f'Maximum difference in the IFCs, averaged over the last {n_rc2_to_average} extractions: {avg_max_diff} eV/Angstrom^2')
            print(f'Average difference in the IFCs, averaged over the last {n_rc2_to_average} extractions: {avg_avg_diff} eV/Angstrom^2')

            if value_to_compare < ifc_diff_threshold:
                text = colored(f'Since {value_to_compare_txt} = {value_to_compare} < {ifc_diff_threshold} (ev/Angstrom^2), ', 'green')
                text += colored(f'the IFCs can be considered converged with rc2 = {rc2} Angstroms!', 'green', attrs=['bold'])
                print(text)
                converged = True
                i_conv = i_rc2
                break

        else:
            if i_rc2 + 1 == 1:
                print(f'Skipping the convergence assessment, as only {i_rc2+1} rc2 value has been done and we need to average over {n_rc2_to_average} extractions')
            else:    
                print(f'Skipping the convergence assessment, as only {i_rc2+1} rc2 values have been done and we need to average over {n_rc2_to_average} extractions')

    if converged == False:
        print_rb(f'The IFC did not converge! You probably need bigger values of rc2!')

    ifcs = np.array(ifcs)
    # let's save the ifcs somewhere, so that they can be analyzed in the future
    expl = 'This pickle file contains the results of the convergence of IFCs w.r.t. rc2. There are three variables:\n1. this explanatory variabe\n'
    expl += '2. the list ifcs of shape (n_rc2s,natoms_unit, natoms_super, 3, 3) in eV/Angst^2\n3. the list of rc2s.'
    results = [expl, ifcs, rc2s[:i_rc2+1]]
    with open(dir.joinpath('conv_rc2_results.pkl'), 'wb') as fl:
        pkl.dump(results, fl)
    
    last_ifc_path = dir.joinpath(f'rc2_{rc2s[i_rc2]}/outfile.forceconstant') # this might be converged or not, but it's the last made
    diffs = np.array([abs(ifcs[i] - ifcs[i-1]) for i in range(1, ifcs.shape[0])])
    max_diffs = np.max(diffs,axis=(1,2,3,4))
    avg_diffs = np.mean(diffs, axis=(1,2,3,4))

    Fig = plt.figure(figsize=(15,4))
    Fig.add_subplot(1,2,1)
    plt.plot(rc2s[1:i_rc2+1], max_diffs, '.')
    plt.title('IFC convergence: max abs. error')
    plt.ylabel('Error on the IFCs (eV/$\mathrm{\AA}^2$)')
    plt.xlabel('rc2 ($\mathrm{\AA}$)')
    
    Fig.add_subplot(1,2,2)
    plt.plot(rc2s[1:i_rc2+1], avg_diffs, '.')
    plt.title('IFC convergence: avg abs. error')
    plt.ylabel('Error on the IFCs (eV/$\mathrm{\AA}^2$)')
    plt.xlabel('rc2 ($\mathrm{\AA}$)')
    
    figpath = dir.joinpath(f'Convergence.png')
    plt.savefig(fname=figpath, bbox_inches='tight', dpi=600, format='png')
    
    return rc2s[i_rc2], i_rc2, last_ifc_path, converged
#    return first_converged, max_diffs, avg_diffs

def conv_rc3_extract_ifcs(unitcell = None,
                          supercell = None,
                          sampling = None,
                          timestep = 1,
                          dir = './',
                          first_order = False,
                          first_order_rc2 = None,
                          displ_threshold_firstorder = 0.0001,
                          max_iterations_first_order = 20,
                          rc2 = None, 
                          rc3s = None, 
                          polar = False,
                          loto_filepath = None, 
                          stride = 1, 
                          temperature = None,
                          bin_prefix = '',
                          tdep_bin_directory = None,
                          ifc_diff_threshold = 0.001, # eV/A^2
                          n_rc3_to_average = 4,
                          conv_criterion_diff = 'avg'): 
    
    """Function to extract the IFCs converging them w.r.t. the rc3 values.
    It is possible to run a first-order TDEP optimisation of the unitcell before extracting the IFCs and it will be done using a rc2
    specific for this stage (`first_order_rc2` or `rc2)` if the first one is None); no rc3 is used in the first-order optimization.
    The same optimised unitcell is then used for all the IFCs extractions with the several rc3 (in `rc3s`).
    The convergence of the IFCs is assessed by evaluating the average differences in the IFCs between the last rc3 value, and the N before,
    and averaging these N average differences (N = `n_rc3_to_average`).  

    Parameters
    ----------
    unitcell : ase.Atoms
        The unit cell.
    supercell : ase.Atoms
        The supercell: must be obtained by repetition of the unitcell via a matrix multiplication!.
    sampling : list or ASE trajectory
        The trajectory used for statistical sampling.
    timestep : float
        Timestep in fs.
    dir : str or Path
        Root directory in which to perform the convergence process.
    first_order : bool
        If True, run a first-order TDEP optimization of the unitcell.
    first_order_rc2 : float or None
        The rc2 cutoff to be used for the first-order optimization.
        If None, `rc2` will be used.
    displ_threshold_firstorder : float
        Convergence threshold for atomic displacements during first-order optimization (in Ang).
    max_iterations_first_order : int
        Maximum number of iterations allowed in the first-order optimization.
    rc2 : list of float
        Value of the second-order cutoff radius (in Ang).
    rc3s: list of float
        List of third-order cutoff radii (in Ang) to test for convergence.
    polar : bool
        Whether to include LO-TO splitting (polar correction).
    loto_filepath : str or Path
        Path to the LO-TO splitting file, required if `polar=True`.
    stride : int
        Stride for sampling frames in TDEP input generation.
    temperature : float
        Temperature (in K).
    bin_prefix : str
        Optional prefix to prepend to TDEP binary calls (e.g. "srun -n 4").
    tdep_bin_directory : str or Path
        Path to the TDEP binaries.
    ifc_diff_threshold : float
        Threshold (in eV/Ang^2) for convergence assessment.
    n_rc3_to_average : int
        Number of previous `rc3` IFCs to average over for convergence assessment.
    conv_criterion_diff : str
        Criterion for convergence: 'avg' for average difference, 'max' for max difference.

    Returns
    -------
    float
        The value of `rc3` (from `rc3s`) at which the IFCs are considered converged.
        If convergence is not reached, returns the last rc3 value tested.

    Raises
    ------
    ValueError
        If input consistency checks fail (e.g., too few rc3s, invalid convergence criterion).
    
    Notes
    -----
    - IFCs are extracted in subfolders named `rc3_<value>` under the main `dir`.
    - A convergence plot (avg/max IFC error vs rc3) is saved as `Convergence.png`.
    - All generated input files are stored in the `infiles` subdirectory.
    - If `first_order=True`, the optimized unitcell is reused for all `rc3` values.
    """
    print_kb('**** IFCs convergence with respect to the 3rd-order cutoff ****')

    dir = Path(dir)
    if not dir.is_dir():
        dir.mkdir(parents=True, exist_ok=True)

    infiles_dir = dir.joinpath('infiles')
    infiles_dir.mkdir(parents=True, exist_ok=True)
    write(infiles_dir.joinpath('infile.ucposcar'), unitcell, format='vasp')
    write(infiles_dir.joinpath('infile.ssposcar'), supercell, format='vasp')
    make_stat(sampling, infiles_dir)
    make_meta(sampling, infiles_dir, timestep=timestep, temp=100)
    make_forces(sampling, infiles_dir)
    make_positions(sampling, infiles_dir)

    if conv_criterion_diff not in ['avg', 'max']:
        raise ValueError('conv_criterion_diff must be either \'avg\' or \'max\'!')
    
    if len(rc3s) <= n_rc3_to_average:
        raise ValueError(f'You asked to assess the convergence by averaging the last {n_rc3_to_average} extractions, but you provided less than {n_rc3_to_average} + 1 values of rc3!')
    
    if first_order == True:
        if first_order_rc2 == None:
            first_order_rc2 = rc2
            print_b('You asked for the first-order TDEP optimization of the unitcell; it will be done using the rc2 you provided.')
        else:
            print_b(f'You asked for the first-order TDEP optimization of the unitcell; it will be done using the rc2 value you provided for this stage ({first_order_rc2} Angstroms).')
        
        fo_dir = dir.joinpath('first_order_optimisation')
        optimised_ucell, optimised_scell, converged = first_order_optimization(from_infiles = False,
                                                   unitcell = unitcell,
                                                   supercell = supercell,
                                                   sampling = sampling,
                                                   timestep = timestep,
                                                   dir = fo_dir,
                                                   displ_threshold_firstorder = displ_threshold_firstorder,
                                                   max_iterations_first_order = max_iterations_first_order,
                                                   rc2 = first_order_rc2,
                                                   polar = polar,
                                                   loto_filepath = loto_filepath,
                                                   stride = stride,
                                                   temperature = temperature,
                                                   bin_prefix = bin_prefix,
                                                   tdep_bin_directory = tdep_bin_directory)
        if converged == False:
            print(colored('The first-order optimization did not converge. However, the last unitcell generated in the process will be used.', 'yellow'))
        unitcell = optimised_ucell
        supercell = optimised_scell
        ln_s_f(fo_dir.joinpath(f'optimized_unitcell.poscar'), infiles_dir.joinpath('infile.ucposcar'))
        ln_s_f(fo_dir.joinpath(f'optimized_supercell.poscar'), infiles_dir.joinpath('infile.ssposcar'))

    print('Starting the extraction for each rc3 value')

    ifcs = []
    for i_rc3, rc3 in enumerate(rc3s):
        print_b(f'+++ rc3 = {rc3} Angstroms +++')
        rc3_dir = dir.joinpath(f'rc3_{rc3}')
        rc3_dir.mkdir(parents=True, exist_ok=True)
        extract_ifcs(from_infiles = True,
                infiles_dir = infiles_dir,
                unitcell = None, # no need
                supercell = None, # no need
                sampling = None, # no need
                timestep = timestep,
                dir = rc3_dir,
                first_order = False,
                displ_threshold_firstorder = None,
                max_iterations_first_order = None,
                rc2 = rc2, 
                rc3 = rc3, 
                polar = polar,
                loto_filepath = loto_filepath, 
                stride = stride, 
                temperature = temperature,
                bin_prefix = bin_prefix,
                tdep_bin_directory = tdep_bin_directory)
        
        new_ifcs = parse_outfile_forceconstants(rc3_dir.joinpath('outfile.forceconstant'), unitcell, supercell) # shape: n_atoms_ucell, n_atoms_scell, 3, 3
        ifcs.append(new_ifcs)  

        converged = False
        if i_rc3 >= n_rc3_to_average:
            print('Assessing the convergence')
            cifcs = np.array(ifcs.copy()) # "c" stands for copy; shape: n_rc3_done, n_atoms_ucell, n_atoms_scell, 3, 3
            diffss = np.abs(cifcs[-1][np.newaxis,:] - cifcs[-1-n_rc3_to_average:-1]) # shape: n_rc3_to_average, n_atoms_ucell, n_atoms_scell, 3, 3
            avg_diffss = np.mean(diffss, axis=(1,2,3,4)) # shape: n_rc3_to_average
            avg_avg_diff = np.mean(avg_diffss)
            max_diffss = np.max(diffss, axis=(1,2,3,4)) # shape: n_rc3_to_average
            avg_max_diff = np.mean(max_diffss)
            
            if conv_criterion_diff == 'avg':
                value_to_compare_txt = 'average'
                value_to_compare = avg_avg_diff
            elif conv_criterion_diff == 'max':
                value_to_compare_txt = 'maximum'
                value_to_compare = avg_max_diff
            
            print(f'Maximum difference in the IFCs, averaged over the last {n_rc3_to_average} extractions: {avg_max_diff} eV/Angstrom^2')
            print(f'Average difference in the IFCs, averaged over the last {n_rc3_to_average} extractions: {avg_avg_diff} eV/Angstrom^2')

            if value_to_compare < ifc_diff_threshold:
                text = colored(f'Since {value_to_compare_txt} = {value_to_compare} < {ifc_diff_threshold} (ev/Angstrom^2), ', 'green')
                text += colored(f'the IFCs can be considered converged with rc2 = {rc2} and rc3 = {rc3} Angstroms!', 'green', attrs=['bold'])
                print(text)
                converged = True
                i_conv = i_rc3
                break

        else:
            if i_rc3 + 1 == 1:
                print(f'Skipping the convergence assessment, as only {i_rc3+1} rc3 value has been done and we need to average over {n_rc3_to_average} extractions')
            else:    
                print(f'Skipping the convergence assessment, as only {i_rc3+1} rc3 values have been done and we need to average over {n_rc3_to_average} extractions')

    if converged == False:
        print_rb(f'The IFC did not converge! You probably need bigger values of rc3!')

    ifcs = np.array(ifcs)
    # let's save the ifcs somewhere, so that they can be analyzed in the future
    expl = 'This pickle file contains the results of the convergence of IFCs w.r.t. rc3. There are three variables:\n1. this explanatory variabe\n'
    expl += '2. the list ifcs of shape (n_rc3s,natoms_unit, natoms_super, 3, 3) in eV/Angst^2\n3. the list of rc3s.'
    results = [expl, ifcs, rc3s[:i_rc3+1]]
    with open(dir.joinpath('conv_rc3_results.pkl'), 'wb') as fl:
        pkl.dump(results, fl)
    
    last_ifc_path = dir.joinpath(f'rc3_{rc3s[i_rc3]}/outfile.forceconstant') # this might be converged or not, but it's the last made
    diffs = np.array([abs(ifcs[i] - ifcs[i-1]) for i in range(1, ifcs.shape[0])])
    max_diffs = np.max(diffs,axis=(1,2,3,4))
    avg_diffs = np.mean(diffs, axis=(1,2,3,4))

    Fig = plt.figure(figsize=(15,4))
    Fig.add_subplot(1,2,1)
    plt.plot(rc3s[1:i_rc3+1], max_diffs, '.')
    plt.title('IFC convergence: max abs. error')
    plt.ylabel('Error on the IFCs (eV/$\mathrm{\AA}^2$)')
    plt.xlabel('rc3 ($\mathrm{\AA}$)')
    
    Fig.add_subplot(1,2,2)
    plt.plot(rc3s[1:i_rc3+1], avg_diffs, '.')
    plt.title('IFC convergence: avg abs. error')
    plt.ylabel('Error on the IFCs (eV/$\mathrm{\AA}^2$)')
    plt.xlabel('rc3 ($\mathrm{\AA}$)')
    
    figpath = dir.joinpath(f'Convergence.png')
    plt.savefig(fname=figpath, bbox_inches='tight', dpi=600, format='png')
    
    return rc3s[i_rc3], i_rc3, last_ifc_path, converged


def parse_outfile_forceconstants(filepath, unitcell, supercell):
    
    def find_index_in_unitcell(red_position, unitcell):
        scaled_positions = unitcell.get_scaled_positions()
        return np.argmin(np.linalg.norm(scaled_positions - red_position, axis=1))

    mat = supercell.get_cell() @ np.linalg.inv(unitcell.get_cell())
    filepath = Path(filepath)
    
    with open(filepath, 'r') as fl:
        lines = fl.readlines()
    lines = [x.split() for x in lines]
    nats_s = len(supercell)
    nats_u = len(unitcell)

    cell = unitcell.get_cell()
    upositions_red = unitcell.get_scaled_positions()# reduced positions in the unit cell, in unit cell coord
    upositions_red[np.abs(upositions_red) < 1E-10] = 0 
    
    positions = supercell.get_positions() # nats, 3
    positions_red = (np.linalg.inv(cell.T) @ positions.T).T # in unit cell coord
    positions_red[np.abs(positions_red) < 1E-10] = 0
    atoms_tuples = []
    for i, atom in enumerate(supercell):
        frac_part = positions_red[i] % 1 # works for both positive and negative reduced coords!!! e.g. -1.3 % 1 = 0.7, not -0.3
        ind = find_index_in_unitcell(frac_part, unitcell)
        repetition_indices = [np.floor(positions_red[i][0]).astype(int), np.floor(positions_red[i][1]).astype(int),np.floor(positions_red[i][2]).astype(int)] # again, works for positive and negative numbers, np.floor(-1.3) = -2, not -1!!
       
        atoms_tuples.append((ind, *repetition_indices))
        
    # now atoms_tuples is a list of tuples (ind, R1, R2, R3), where ind the index of the atom in the unitcell and R1/2/3 are the component of the position of the repetition in reduced coordinates
    

    ifc = np.zeros((nats_u, nats_s, 3, 3))
    k = 2
    for i in range(nats_u):
        nns = int(lines[k][0]) # number of neighbours
        for n in range(nns):
            ci = k+1+5*n
            neigh_unit_ind = int(lines[ci][0])-1 # index of the neighbour in the unitcell
            
            # let's retrieve the vector of the repetition of the unitcell of the current neighbour in the unit cell (red) coords.
            vec_u = np.array([float(lines[ci+1][0]), float(lines[ci+1][1]), float(lines[ci+1][2])])

            # let's get the the vector of the the current neighbour in the unit cell (red) coords.
            curr_pos_u = upositions_red[neigh_unit_ind] + vec_u 

            # let's write curr_pos_u in the super cell (red) coords            
            # we need to change the basis of from the unitcell vectors to the supercell vectors
            # x^u = L x^s where L is P.T, where P is the mat_mult used to create the supercell from the unitcell
            # in principle we can compute x^s = L^-1 @ x^u, but np.solve is faster
            
            curr_pos_s = np.linalg.solve(mat.T, curr_pos_u) # position of the current neighbour in the super cell (red) coords.
            curr_pos_s[np.abs(curr_pos_s) < 1E-10] = 0 # clean
            
            # let's wrap it inside the supercell
            curr_pos_s_wrapped = curr_pos_s % 1 

            # now let's put it back to the unit cell (red) coords.
            curr_pos_u_wrapped = np.linalg.solve(np.linalg.inv(mat.T), curr_pos_s_wrapped)

            # let's write the vector of the repetition of the unitcell of the current neighbour in the unit cell (red) coords. after the wrapping
            curr_repetition_indices = [np.floor(curr_pos_u_wrapped[0]).astype(int), np.floor(curr_pos_u_wrapped[1]).astype(int),np.floor(curr_pos_u_wrapped[2]).astype(int)] 
            
            current_tuple = (neigh_unit_ind, curr_repetition_indices[0], curr_repetition_indices[1], curr_repetition_indices[2])
            j = atoms_tuples.index(current_tuple) # find the matching tuple, j is the index of the neighbour in the supercell
            tens = []
            tens.append([float(lines[ci+2][0]), float(lines[ci+2][1]), float(lines[ci+2][2])])
            tens.append([float(lines[ci+3][0]), float(lines[ci+3][1]), float(lines[ci+3][2])])
            tens.append([float(lines[ci+4][0]), float(lines[ci+4][1]), float(lines[ci+4][2])])
            tens = np.array(tens, dtype='float')
            ifc[i,j] += tens
        k += 1+5*nns

    return ifc # shape n_atoms_unitcell, n_atoms_supercell, 3, 3

def run_phonons(dir, ucell, scell, ifc_file, qgrid=32, tdep_bin_directory=None, bin_pref='', dos=False, units='thz'):
    write(dir.joinpath('infile.ucposcar'), ucell, format='vasp')
    write(dir.joinpath('infile.ssposcar'), scell, format='vasp')
    shutil.copy(ifc_file, dir.joinpath('infile.forceconstant'))

    if tdep_bin_directory is None or tdep_bin_directory == '':
        tdep_bin_directory = g_tdep_bin_directory
    else:
        tdep_bin_directory = Path(tdep_bin_directory) # if it's already a Path(), this won't change anything

    if dos == False:
        dos = ''
    else:
        dos = '--dos'
    cmd = f'{bin_pref} {tdep_bin_directory.joinpath("phonon_dispersion_relations")} -qg {qgrid} {qgrid} {qgrid} {dos} --unit {units}'
    logpath = dir.joinpath('log_ph')
    errpath = dir.joinpath('err_ph')
    print(cmd)
    with open(logpath.absolute(), 'w') as log, open(errpath.absolute(), 'w') as err:
            run(cmd.split(), cwd=dir.absolute(), stdout=log, stderr=err)

def first_order_optimization(from_infiles=False,
                             infiles_dir=None,
                            unitcell = None,
                            supercell = None,
                            sampling = None,
                            timestep = 1,
                            dir = './',
                            displ_threshold_firstorder = 0.0001,
                            max_iterations_first_order = 20,
                            rc2 = None, 
                            polar = False,
                            loto_filepath = None, 
                            stride = 1, 
                            temperature = None,
                            bin_prefix = '',
                            tdep_bin_directory = None):
    """Perform first-order TDEP optimization of atomic positions in the unit cell.

    This function iteratively refines the atomic positions of the unit cell using 
    first-order force constants extracted via TDEP, until the maximum atomic displacement 
    between iterations falls below a given threshold (`displ_threshold_firstorder`), 
    or until the maximum number of iterations is reached.

    Parameters
    ----------
    from_infiles : bool
        If True, use input files from `infiles_dir`. Otherwise, use provided `unitcell`, 
        `supercell`, and `sampling` data.
    infiles_dir : str or Path
        Path to directory containing the TDEP input files, required if `from_infiles` is True.
    unitcell : ase.Atoms
        Initial unit cell structure.
    supercell : ase.Atoms
        Initial supercell structure.
    sampling : list of ase.Atoms
        List of trajectory frames used to extract forces and displacements.
    timestep : float
        Time step in femtoseconds used in the MD sampling (anything is fine if it's sTDEP).
    dir : str or Path
        Directory in which to perform the optimization and store intermediate files.
    displ_threshold_firstorder : float
        Convergence threshold for maximum atomic displacement (in Å).
    max_iterations_first_order : int
        Maximum number of optimization iterations.
    rc2 : float
        Second-order interaction cutoff radius (in Å).
    polar : bool
        Whether to include LO-TO splitting corrections.
    loto_filepath : str or Path
        Path to the `infile.lotosplitting` file (required if `polar` is True).
    stride : int
        Stride for reading the MD sampling trajectory.
    temperature : float
        Temperature (in K).
    bin_prefix : str
        Optional prefix for the TDEP binary call (e.g., to run with MPI).
    tdep_bin_directory : str or Path
        Path to the directory containing the TDEP binaries.

    Returns
    -------
    conv_unitcell : ase.Atoms
        The optimized unit cell.
    conv_supercell : ase.Atoms
        The supercell generated from the optimized unit cell.

    Raises
    ------
    RuntimeError
        If the structure does not converge within the allowed number of iterations.
    TypeError
        If required arguments are missing or inconsistent.
    """
    
    # FUNCTIONS 
    
    def check_convergence_positions(dir, displ_threshold_firstorder):
        old_uc = read(dir.joinpath('infile.ucposcar'), format='vasp')
        old_positions = old_uc.get_positions()
        old_red_positions = old_uc.get_scaled_positions()
        new_uc = read(dir.joinpath('outfile.new_ucposcar'), format='vasp')
        new_positions = new_uc.get_positions()
        new_red_positions = new_uc.get_scaled_positions()
        err = new_positions - old_positions
        red_err = new_red_positions - old_red_positions

        if (err >= displ_threshold_firstorder).any():
            return False, err, red_err
        else:
            return True, err, red_err
    
    def move(src, dest):
        at = read(src, format='vasp')
        write(dest, at, format='vasp')
        src.unlink()
    
    ############

    print_kb('**** First-order TDEP optimization of the unitcell ****')
    dir = Path(dir)
    if dir.is_dir() and not dir.resolve().absolute() == Path('./').resolve().absolute():
        shutil.rmtree(dir)
    dir.mkdir(parents=True, exist_ok=True)


    if from_infiles == True:
        if infiles_dir is None:
            raise TypeError('Since from_infiles is True, you must provide infiles_dir!')
        else:
            src_infiles_dir = Path(infiles_dir)
    else:
        if any([unitcell is None, supercell is None, sampling is None]):
            raise TypeError('Since from_infiles is False, you must provide the unitcell, the supercell and the sampling trajectory!')

    if rc2 is None:
        raise TypeError('rc2 is mandatory!')
    else:
        rc2_cmd = f'-rc2 {rc2}'

    
    if polar == True:
        if loto_filepath is None:
            raise TypeError('Since polar is True, you must provide loto_filepath!')
        else:
            polar_cmd = '--polar'
            loto_filepath = Path(loto_filepath)
    else:
        polar_cmd = ''

    if tdep_bin_directory is None or tdep_bin_directory == '':
        tdep_bin_directory = g_tdep_bin_directory
    else:
        tdep_bin_directory = Path(tdep_bin_directory) # if it's already a Path(), this won't change anything
    
    stride_cmd = f'--stride {stride}'

    temperature_cmd = f'--temperature {temperature}'



    #infiles_dir = Path(dir.joinpath('infiles'))
    #infiles_dir.mkdir(parents=True, exist_ok=True)
    if from_infiles == True:
        ln_s_f(src_infiles_dir.joinpath('infile.ucposcar'), dir)
        ln_s_f(src_infiles_dir.joinpath('infile.ssposcar'), dir)
        ln_s_f(src_infiles_dir.joinpath('infile.meta'), dir)
        ln_s_f(src_infiles_dir.joinpath('infile.stat'), dir)
        ln_s_f(src_infiles_dir.joinpath('infile.positions'), dir)
        ln_s_f(src_infiles_dir.joinpath('infile.forces'), dir)
        unitcell = read(src_infiles_dir.joinpath('infile.ucposcar'), format='vasp')
        supercell = read(src_infiles_dir.joinpath('infile.ssposcar'), format='vasp')
    else:       
        write(dir.joinpath('infile.ucposcar'), unitcell, format='vasp')
        write(dir.joinpath('infile.ssposcar'), supercell, format='vasp')
        make_stat(sampling, dir)
        make_meta(sampling, dir, timestep=timestep, temp=100)
        make_forces(sampling, dir)
        make_positions(sampling, dir)

    if polar == True:
        ln_s_f(loto_filepath, dir.joinpath('infile.lotosplitting'))
    
    cmd = f'{bin_prefix} {tdep_bin_directory.joinpath("extract_forceconstants")} --firstorder {rc2_cmd} {temperature_cmd} {polar_cmd} {stride_cmd}'

    ucells = [unitcell]
    mat = np.round(supercell.get_cell() @ np.linalg.inv(unitcell.get_cell())) # rounding is necessary, otherwise astype will truncate
    mat = mat.astype('int')
    
    print('Here are the parameters of the first-order optimization:')
    print(f'\t2nd-order cutoff: ' +colored(f'{rc2} Angstroms', 'blue'))
    print(f'\tTemperature: ' +colored(f'{temperature}', 'blue'))
    if bin_prefix == '':
        print(f'\tNo binary prefix')
    else:
        print(f'\tBinary prefix: ' +colored(f'{bin_prefix}', 'blue'))
    if polar == True:
        print('\tPolar correction (LO-TO): ' + colored('yes', 'blue'))
    else:
        print('\tPolar correction (LO-TO): ' + colored('no', 'blue'))
    print(f'\tStride: ' +colored(f'{stride}', 'blue'))
    print(f'\tMaximum number of iterations: ' +colored(f'{max_iterations_first_order}', 'blue'))
    print(f'\tThreshold for the maximum difference in the positions between iterations: ' +colored(f'{displ_threshold_firstorder} Angstrom', 'blue'))
    print(f'\tStarting reduced positions:')
    for pos in unitcell.get_scaled_positions():
        print(f'\t                   ' +colored(f'{pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}', 'blue'))
    print('\tMultiplicity matrix:')
    print(f'\t                    ' +colored(f'{mat[0,0]} {mat[0,1]} {mat[0,2]}', 'blue'))
    print(f'\t                    ' +colored(f'{mat[1,0]} {mat[1,1]} {mat[1,2]}', 'blue'))
    print(f'\t                    ' +colored(f'{mat[2,0]} {mat[2,1]} {mat[2,2]}', 'blue'))
    print(f'-- Starting with iterations --')
    
    logpath = dir.joinpath('log_ifc')
    errpath = dir.joinpath('err_ifc')

    for i in range(max_iterations_first_order):
        print(f'# Iteration {i+1}:')
        with open(logpath.absolute(), 'w') as log, open(errpath.absolute(), 'w') as err:
            run(cmd.split(), cwd=dir.absolute(), stdout=log, stderr=err)
        ucells.append(read(dir.joinpath('outfile.new_ucposcar'), format='vasp'))
        converged, err, red_err = check_convergence_positions(dir, displ_threshold_firstorder)
        dir.joinpath('infile.ucposcar').rename(dir.joinpath(f'infile.ucposcar_{i}')) # rename the ucell that's been used with the previous iteration number
        move(dir.joinpath('outfile.new_ucposcar'), dir.joinpath('infile.ucposcar'))
        new_unitcell = ucells[-1]
        new_supercell = make_supercell(new_unitcell, mat)
        dir.joinpath('infile.ssposcar').rename(dir.joinpath(f'infile.ssposcar_{i}'))
        write(dir.joinpath('infile.ssposcar'), new_supercell, format='vasp')
        dir.joinpath('outfile.new_ssposcar').unlink()

        # now infile.ucposcar is the newly generated one
        
        print('\tNew reduced positions:')
        for pos in new_unitcell.get_scaled_positions():
            print(f'\t                   ' + colored(f'{pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}', 'blue'))

        print(f'\tMaximum difference in position: ' + colored(f'{err.max():.12f} Angstrom ({red_err.max():.12f} in reduced units)', 'blue'))
        
        # let's save the last unitcell and supercell as 'optimized' regardless of it being converged or not
        ln_s_f(dir.joinpath(f'infile.ucposcar_{i+1}'), dir.joinpath('optimized_unitcell.poscar')) # create a sym link to the last unitcell
        ln_s_f(dir.joinpath(f'infile.ssposcar_{i+1}'), dir.joinpath('optimized_supercell.poscar')) # create a sym link to the last supercell
        if converged:
            conv_unitcell = new_unitcell
            conv_supercell = new_supercell 
            dir.joinpath('infile.ucposcar').rename(dir.joinpath(f'infile.ucposcar_{i+1}')) # change the name of the converged infile consistently with the previous ones
            dir.joinpath('infile.ssposcar').rename(dir.joinpath(f'infile.ssposcar_{i+1}'))
            #print(f'renamed: infile.ucposcar to infile.ucposcar_{i+1}')
            
            print_gb(f'Convergence reached at iteration {i+1} ({err.max():.12f} < {displ_threshold_firstorder:.12f} Angstroms)!')
            print(f'Final reduced positions:')
            for pos in conv_unitcell.get_scaled_positions():
                print(f'\t                   ' + colored(f'{pos[0]:.12f} {pos[1]:.12f} {pos[2]:.12f}', 'blue'))
            print(f'Difference with respect to the previous iteration:')
            for diff in red_err:
                print(f'\t                    '+ colored(f'{diff[0]:.12f} {diff[1]:.12f} {diff[2]:.12f}', 'blue'))
            print('The optimised unitcell is saved as ' + colored('optimized_unitcell.poscar', 'blue', attrs=['bold']) + '.')  
            print('The optimised supercell is saved as ' + colored('optimized_supercell.poscar', 'blue', attrs=['bold']) + '.') 
            print_kb('********* ' + colored('First-order TDEP optimization done!', 'green') + ' *********') 
            break

        if i == max_iterations_first_order - 1:
            dir.joinpath('infile.ucposcar').rename(dir.joinpath(f'infile.ucposcar_{i+1}')) # change the name of the last infile consistently with the previous ones
            dir.joinpath('infile.ssposcar').rename(dir.joinpath(f'infile.ssposcar_{i+1}')) 
            print_r('ATTENTION! The maximum number of iterations was reached without convergence!')
            print_r('The optimised unconverged unitcell is saved as ' + colored('optimized_unitcell.poscar', 'blue', attrs=['bold']) + '.')  
            print_r('The optimised unconverged supercell is saved as ' + colored('optimized_supercell.poscar', 'blue', attrs=['bold']) + '.')
            print_rb('******** First-order TDEP optimization failed! ********')

    return new_unitcell, new_supercell, converged



def correct_spectralfunction(sp_path):
    '''Correct the spectralfunction coming from TDEP `lineshape` binary
    Return
    '''
    with h5py.File(sp_path) as file:
        Delta = np.array(file['anharmonic']['real_threephonon_selfenergy'])[3:] # shape: nmodes, n_freqs
        Delta_0 = Delta[:,0]
        Gamma = np.array(file['anharmonic']['imaginary_threephonon_selfenergy'])[3:] # shape: nmodes, n_freqs
        isotope = np.array(file['anharmonic']['imaginary_isotope_selfenergy'])[3:] # shape: nmodes, n_freqs
        Gamma = Gamma + isotope
        lns = np.array(file['anharmonic']['spectralfunction_per_mode'])[3:]       
        probe_freqs = np.array(file['anharmonic']['frequency']) # shape: n_freqs
        ph_freqs = np.array(file['harmonic']['harmonic_frequencies'])[3:] # shape: nmodes 
    cs_numerator = 2 * ph_freqs[:,None] * Gamma * 2 * ph_freqs[:,None]/np.pi
    cs_denominator = (probe_freqs[None,:]**2 - ph_freqs[:,None]**2 - 2 * ph_freqs[:,None] * (Delta-Delta_0[:,None]))**2 + 4 * ph_freqs[:,None]**2 * Gamma**2  
    cs = cs_numerator / cs_denominator
    return cs, probe_freqs      
