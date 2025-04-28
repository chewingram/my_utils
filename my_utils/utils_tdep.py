import numpy as np
from ase.io import read, write
from ase.build import make_supercell
import sys
from .utils import from_list_of_numbs_to_text, data_reader, ln_s_f
import os
from subprocess import run
from math import floor, ceil
import h5py
from matplotlib import pyplot as plt
from pathlib import Path
import shutil
from copy import deepcopy as cp

binpath = shutil.which('extract_forceconstants')
if binpath is not None:
    g_tdep_bin_directory = Path(binpath).absolute()
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
            confs.append(cp(conf))
            
        except: # e.g. the file is bugged/corrupted
            pass

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

def make_canonical_configurations(nconf, temp, quantum, dir, outfile_name, max_freq=False, pref_bin='', tdep_bin_directory=None):
    '''
    Function to launch tdep canonical-configurations specifying only the number of confs, the temperature and
    the quantum flag. After this, all (non bugged) files are merged in a single ASE traj (outfile_name).
    If canonical-sampling produces some bugged confs, it will be regenerated and saved in the output file.
    Args:
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
    
    if max_freq != False:
        max_freq = f'--maximum_frequency {max_freq}'
    else:
        max_freq = ''
            
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

def convergence_tdep_mdlen(
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


    