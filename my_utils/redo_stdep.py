import numpy as np
import os
import shutil
from pathlib import Path
from matplotlib import pyplot as plt

from my_utils import utils_mlip as mlp
from my_utils import utils_tdep as tdp
from my_utils.utils import min_distance_to_surface, ln_s_f
from ase.io import read, write
from ase.build import make_supercell as mk_supcercell

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
    rc3: float = None,
    ifc_max_err_threshold: float = None,
    niters: int = 10,
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

    rc3 : float, optional
        Third-order interaction cutoff (in Å). May be unused depending on setup.

    ifc_max_err_threshold : float, optional
        Maximum allowed error threshold (in eV/Å²) to consider the IFCs converged at each iteration.
        Default if 0.0001.

    niters : int, optional
        Total number of sTDEP iterations. Default is 10.

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
            scell = mk_supcercell(ucell, scell_mat)
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
    nconfs = [4]
    nconfs.extend([x*20 for x in range(1,7)])

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
        # print('min dist')
        # print(min_dist)
        rc2s = [x for x in range(int(max(min_dist - 10, 0)), int(min_dist+1))] #
        rc2s = [4,5,6,7,8,9]
        print(f'rc2s: {rc2s}')
        
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
                                                                        polar = False,
                                                                        stride = 1, 
                                                                        temperature = T,
                                                                        bin_prefix = pref_bin,
                                                                        tdep_bin_directory = tdep_bin_directory,
                                                                        max_err_threshold = ifc_max_err_threshold)
        
        ln_s_f(last_ifc_path, iter_dir.joinpath('converged_outfile.forceconstant'))
        print(f'The converged one is {last_ifc_path}')
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
    print(ifcs.shape)
    ifcs = np.array(ifcs)
    diffs = np.abs(ifcs[1:] - ifcs[:-1])
    max_diffs = np.max(diffs, axis=(1, 2, 3, 4))
    print(max_diffs)
    Fig = plt.figure(figsize=(15,4))
    plt.plot(nconfs[1:], max_diffs, '.')
    plt.title('IFC convergence: max abs. error')
    plt.ylabel('Error on the IFCs (eV/$\mathrm{\AA}^2$)')
    plt.xlabel('Number of structures')
    figpath = root_dir.joinpath(f'Convergence.png')
    plt.savefig(fname=figpath, bbox_inches='tight', dpi=600, format='png')

    



