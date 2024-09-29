
class MlipCalculator():
    def __init__(self, initial_train_set=None):
        if initial_train_set != None:
            if isinstance(initial_train_set, type(ase.atoms.Atoms())):
                self.train_set = [initial_train_set]
            elif isinstance(initial_train_set, list):
                assert all([isinstance(x, type(ase.atoms.Atom())) for x in initial_train_set]), f"initial_train_set must be an (list of) ase Atoms object(s)"
                self.train_set = initial_train_set.copy()
            else:
                raise TypeError(f"initial_train_set must be an (list of) ase Atoms object(s)")
    @abc.abstractmethod
    def update_train_number(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def add_structures_to_train_set(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def add_structures_and_train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train_potential(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def calc_efs(self, *args, **kwargs):
        pass
    


class MtpMlipCalculator(MlipCalculator):
    '''
    '''
    def __init__(self,
                 bin_path,
                 preexists=False,
                 preexisting_pot_path=None,
                 untrained_mtp_dir=None,
                 mtp_level=None,
                 root_dir='./MTP', # root directory OF THE MTP CALCULATOR
                 initial_train_set=None,
                 keep_last_train_set=True,
                 paral_command='',
                 training_params=None):
        '''
        Parameters
        ----------
        bin_path: str
            path to the MTP binary file
        preexists: bool, default=False
            - True: a preexisting trained MTP model will be loaded     
                    (see parameter 'preexisting_pot_path')
            - False: a new model will be trained
        preexisting_pot_path: str, default=None
            mandatory when preexists=True, ignored otherwise; path
            to the .mtp file containing the preexisting trained 
            potential
        untrained_mtp_dir: str
            path to the directory containing all the untrained .mtp       
            files
        mtp_level: {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 24, 26, 28}
            level of the mtp model to train
        root_dir: str
            path to the root directory of the mlip calculator
        initial_train_set: ase.atoms.Atoms or list of ase.atoms.Atoms
            (list of) ase Atoms object(s) to set as dataset before any       
            MLACS iteration is done; it will remain part of the dataset
        keep_last_train_set: bool
            - True: the dataset used for the last training is kept
                    in the directory of trained potential of the mlip
                    object as as a list of ase Atoms object called
                    "last_dataset.traj"
        paral_command: str
            command to use before calling binaries; this is meant to
            be used for parallelisation (e.g. mpirun, srun), but can
            be used for any purpose, if needed
        training_params: dict 
            dictionary containing the flags to use in the training; 
            these are the possibile flags:
            ene_weight: float, default=1
                weight of energies in the fitting
            for_weight: float, default=0.01
                weight of forces in the fitting
            str_weight: float, default=0.001 
                weight of stresses in the fitting
            sc_b_for: float, default=0
                if >0 then configurations near equilibrium (with 
                roughly force < <double>) get more weight
            val_cfg: str 
                filename with configuration to validate
            max_iter: int, default=1000
                maximal number of iterations
            cur_pot_n: str
                if not empty, save potential on each iteration with 
                name = cur_pot_n
            tr_pot_n: str, default=Trained.mtp
                filename for trained potential.
            bgfs_tol: float, default=1e-3
                stop if error dropped by a factor smaller than this 
                over 50 BFGS iterations
            weighting: {'vibrations', 'molecules', 'structures'}, default=vibrations 
                how to weight configuration wtih different sizes 
                relative to each other
            init_par: {'random', 'same'}, default='random'
                how to initialize parameters if a potential was not 
                pre-fitted;
                - random: random initialization
                - same: this is when interaction of all species is 
                        the same (more accurate fit, but longer 
                        optimization)
            skip_preinit: bool 
                skip the 75 iterations done when parameters are not 
                given
            up_mindist: bool
                updating the mindist parameter with actual minimal 
                interatomic distance in the training set
        '''
        
        super.__init__(initial_train_set) # set initial_train_set if exists
        
        self.bin_path = Path(bin_path)
        
        assert self.bin_path.is_file(), f"bin_path does not exists, or "\
                                      + f"is not a regular file!"

        self.is_trained = False
        
        # chek (if) preexisting potential 
        if preexists == True:
            assert preexisting_pot_path != None, f"when preexists=True "\
                + f"preexisting_pot_path must be given!"
            preexisting_pot_path = Path(preexisting_pot_path)
            assert preexisting_pot_path.is_file, \
            f"{preexisting_pot_path.absolute} is not a file!"
            self.set_trained_pot_path(preexisting_pot_path)
            self.is_trained = True
        else: # we are creating a new mlip calculator
                # in this case untrained_mtp_dir cannot be None
            assert untrained_mtp_dir != None, f"You are creating a new"\
                + f"MtpMlipCalculator object and the directory containing"\
                + f" the untrained .pot files must be given"
            assert isinstance(untrained_mtp_dir, str), f"untrained_mtp_dir "\
                + f"must be a string!"
            
            # set the untrained_mtp_dir
        if isinstance(untrained_mtp_dir, str):
            untrained_mtp_dir = Path(untrained_mtp_dir)
        elif untrained_mtp_dir == None:
            self.untrained_mtp_dir = None
            
        assert isinstance(mtp_level, int) or isinstance(mtp_level, float), \
            f"mtp_level must be a integer!"
        
        self.mtp_level = mtp_level
        
        self.training_params = training_params
        
        self.root_dir = Path(root_dir)
        
        if not self.root_dir.is_dir():
            os.makedirs(self.root_dir)
        
        self.update_train_number()
        
        self.paral_command = paral_command            

    
    def set_trained_pot_path(self, path):
        '''Function to set the current trained potential file path
        Given the path of a file, it is renamed "pot.mtp" and copied
        in 
        '''
        if not isinstance(path, Path()):
            path = Path(path)
        self.trained_pot_path = path

    
    def update_train_number(self):
        '''Function to update the number of structures in the training set
        
        It looks how many structures are in self.train_set and update 
        self.train_number
        
        '''
        self.train_number = len(self.train_set)
    
    
    def add_structures_to_train_set(self, structures):
        '''Function to add one or more structures to the training set
        
        Parameters
        ----------
        structures : ase.atoms.Atoms or list of ase.atoms.Atoms
            structures to add to the training set
            
        '''
        if isinstance(initial_trainset, type(ase.atoms.Atoms())):
            self.trainset.append(structures)
        elif isinstance(initial_trainset, list):
            assert all([isinstance(x, type(ase.atoms.Atom())) for x in structures]), f"structures must be an (list of) ase Atoms object(s)"
            self.trainset.extend(structures)
        else:
            raise TypeError(f"structures must be an (list of) ase Atoms object(s)")
        for structure in structures:
            try:
                structure.get_potential_energy()
            except:
                raise ValueError(f"One or more structures don't have computed properties stored!")
        self.update_train_number()
        
        
    def add_structures_and_train(self, structures, wdir=None):
        '''Function to add structures to the training set and launch
           the training
           
           Parameters
           ----------
           wdir: str
               path to the directory where to run the training and save everything
           structures: ase.atoms.Atoms or list of ase.atoms.Atoms
               structures to add to the training set

        '''
        
        self.ass_structures_to_train_set(structures)
        self.train_potential(self, wdir)
        
        
    def train_potential(self, wdir=None, exclude_indices=None):
        '''Function to train the potential
        
        Parameters
        ----------
        wdir: str, default={self.root_dir}/training/
            directory where to run the training and save everything
                                      
        '''
        
        if wdir == None:
            wdir = self.root_dir.joinpath('training')
        wdir = Path(wdir)
        
        if wdir.is_dir() == False:
            os.makedirs(wdir)
            
        if exclude_indices == None:
            exclude_indices = []
        train_structures = [self.train_set[x] for x in range(self.train_number)\
                            if x not in exclude_indices]
        
        train_pot_from_ase_tmp(mpirun=self.paral_command, 
                               mlip_bin=self.bin_path,
                               untrained_pot_file_dir=self.untrained_pot_file_dir,
                               mtp_level=self.mtp_level,
                               train_set=train_structures, 
                               dir=wdir, 
                               params=self.training_params)
        # we need something here to set the current trained pot file and 
        # "trained" status of the object
    
    
    def calc_efs(self, wdir, structures, save=False, file_name=None):
        '''Function to compute energy, forces and stresses of one or more structures
        
        Parameters
        ----------
        ########################################################################
        wdir: str
            path to the directory where to run the calculation and (eventually)
            save everything
        structures: ase.atoms.Atoms or list of ase.atoms.Atoms
            structures to add to the training set
        save: bool
            - True: an ase trajectory will be saved with the structures and 
                    their properties
            - False: no saving
        file_name: str
            mandatory when 'save'=True; name of the trajectory to save; it will 
            be saved inside 'wdir'
        
        Returns
        -------
        structures: list of ase.atoms.Atoms
            trajectory containing the same input structures with their 
            calculated properties stored
            
        '''
        
        if isinstance(initial_trainset, type(ase.atoms.Atoms())):
            self.trainset.append(structures)
        elif isinstance(initial_trainset, list):
            assert all([isinstance(x, type(ase.atoms.Atom())) for x in structures]), \
                   f"structures must be an (list of) ase Atoms object(s)"
            self.trainset.extend(structures)
        else:
            raise TypeError(f"structures must be an (list of) ase Atoms object(s)")
        
        calc_efs_params = dict(mlip_bin=self.mlip_bin,
                               atoms=structures,
                               mpirun=self.paral_command,
                               pot_path=self.trained_pot_filepath,
                               cfg_files=False,
                               out_path=False,
                               dir=wdir)
        if save == True:
            assert file_name != None, f"Since save=True, file_name must be given!"
            assert isinstance(file_name, str), f"file_name must be a string!"
            calc_efs_params['write_conf'] = True
            calc_efs_params['outconf_name'] = file_path
        else:
            calc_efs_params['write_conf'] = False
            
        return calc_efs_from_ase(**calc_efs_params)
    

    def make_comparison(self,
                        structures1=None, 
                        structures2=None,
                        props='all', 
                        units=None):
        
        '''Create the comparison files for energy, forces and stress starting from the .cfg files.
        
        Parameters
        ----------
        structures1: ase.atoms.Atoms or list of ase.atoms.Atoms
            mandatory when is_ase1 = True (ignored otherwise); (list of) 
            ase Atoms object(s) with the true values
        structures2: ase.atoms.Atoms or list of ase.atoms.Atoms
            mandatory when is_ase2 = True (ignored otherwise); (list of) 
            ase Atoms object(s) with the ML values
        props: str or list of {'energy', 'forces', 'stress', 'all'}
            if a list is given containing 'all', all three properties will
            be considered, independent on the other elements of the list
        units: dict, default: {'energy': 'eV/at', 'forces':'eV/Angs', 'stress':'GPa'}
            dictionary with key-value pairs like prop-unit with prop in 
            ['energy', 'forces', 'stress'] and value being a string with 
            the unit to print for the respective property. If None, the 
            respective units will be eV/at, eV/Angs and GPa
    
        Returns
        -------
        errs: list of float
            [rmse, mae, R2] 
            
        '''

        return make_comparison(self,
                               is_ase1=True,
                               is_ase2=True,
                               structures1=structures1,
                               structures2=structures2,
                               file1=None,
                               file2=None,
                               props='all',
                               make_file=False,
                               dir=None,
                               outfile_pref=None,
                               units=None)
        
           ########################################################################

        
        