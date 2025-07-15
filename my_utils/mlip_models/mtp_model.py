import json
import traceback
from copy import deepcopy as dc
from ..utils_mlip import train_pot_from_ase_tmp as train_pot_from_ase, conv_ase_to_mlip2_text, calc_efs_from_ase
from ..mtp_models import MlipModel
from ..utils import flatten



class MTP_model(MlipModel):
    self.all_hyperparameters_names = ['mlip_bin',
                                    'untrained_pot_file_dir',
                                    'mtp_level',
                                    'min_dist',
                                    'max_dist',
                                    'radial_basis_type',
                                    'radial_basis_size',
                                    'ene_weight', 
                                    'for_weight',
                                    'str_weight',
                                    'sc_b_for',
                                    'val_cfg',
                                    'max_iter',
                                    'cur_pot_n', 
                                    'trained_pot_name', 
                                    'bfgs_tol', 
                                    'weighting', 
                                    'init_par', 
                                    'skip_preinit', 
                                    'up_mindist']
    
    self.mandatory_hyperparameters = ['mlip_bin',
                                    'untrained_pot_file_dir',
                                    'mtp_level',
                                    'min_dist',
                                    'max_dist',
                                    'radial_basis_type',
                                    'radial_basis_size']
    
    hyperparameters_names_for_training_function = []
    def __init__(self,
                 root_dir,
                 hyperparameters=None,
                 train_set=None):
        '''
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
        '''
        super.__init__(self, root_dir, hyperparameters=hyperparameters, train_set=train_set)
        self.hyperparameters = dict()
        if hyperparameters is not None:
            for hp_n, hp_v in hyperparameters.items():
                if hp_n in all_hyperparameters_names:
                    self.hyperparameters[hp_n] = hp_v

        
        
        
        # initialize training parameters dictionary
        
        for par in training_params_names:
            self.training_params[par] = None

        # load training parameters from input
        if train_params is not None:
            for train_par_name, train_par_val in train_params.items():
                if train_par_name in self.training_params.keys():
                    self.trainining_params[train_par_name] = train_par_val



    
    def train(self, bin_pref=''):'''
        mlip_bin: str
            path to the MTP binary
        
        untrained_pot_file_dir: str 
            path to the directory containing the untrained mtp init files (.mtp)

        training_dir: str
            path to the directory where to run the training (and save the output)

        bin_pref: str
            command for mpi or similar (e.g. 'mpirun')
        
        final_evaluation: bool
            execute a final evaluation of the potential on the training set
        
        train_set: list of ase.Atoms objects
            if not None, it will be used for the training and any preexisting self.train_set will be overwritten
            if None, self.train_set must not be None

        params: dict 
            dictionary containing the flags to use; these are the possibilities:
            ene_weight: float, default=1
                weight of energies in the fitting
            for_weight: float, default=0.01
                weight of forces in the fitting
            str_weight: float, default=0.001 
                weight of stresses in the fitting
            sc_b_for: float, default=0
                if >0 then configurations near equilibrium (with rougthly force < 
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
                how to weight configuration with different sizes relative to each 
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

        pass
        # todo: check that self.hyperparameters contains the mandatory parameters + the others mandatory vars are there
        # todo: create the params dictionary

        
        results = train_pot_from_ase_tmp(mlip_bin=self.hyperparameters['mlip_bin'],
                                         untrained_pot_file_dir=self.hyperparameters['untrained_pot_file_dir'],
                                         mtp_level=self.hyperparameters['mtp_level'],
                                         min_dist=self.hyperparameters['min_dist'],
                                         max_dist=self.hyperparameters['max_dist'],
                                         radial_basis_size=self.hyperparameters['radial_basis_size'],
                                         radial_basis_type=self.hyperparameters['radial_basis_type'],
                                         train_set=self.train_set,
                                         dir=self.training_dir,
                                         params=params,
                                         mpirun=bin_pref,
                                         final_evaluation=False)
        
        # we need to check that the training went to completion
        with open(training_dir,joinpath('log_train'), 'r') as fl:
            lines = fl.readlines()
            if any(['* * * TRAIN ERRORS * * *' in line for line in lines]):
                success = True
                
            else:
                success = False
                
                return success

        self.species_count = len(set(flatten([x.get_chemical_symbols() for x in train_set])))

        # todo: deal with the final evaluation, if needed
        if final_evaluation == True:
            trained_pot_file_path = results[0]
            ml_trainset = results[1]
            errs = results[2]
            self.ml_trainset = ml_trainset
        else:
            trained_pot_file_path = results
        self.trained_pot = dict()    
        self.trained_pot['path'] = trained_pot_file_path
        with open(trained_pot_file_path, 'r') as fl:
            self.trained_pot['text_file'] = fl.readlines()


    def _compute_properties(atoms, wdir, **kwargs)
            
        if isinstance(atoms, list):
            if not all([isinstance(x, Atoms) for x in list]):
                raise TypeError('The variable `atoms` must be an ASE atoms object or a list of ASE atoms objects!')
        elif isinstance(atoms, Atoms):
            atoms = [atoms]
        
        wdir = Path(wdir)

        if 'bin_pref' in kwargs.keys():
            bin_pref = kwargs['bin_pref']
        else:
            bin_pref = ''

        atoms_calc = calc_efs_from_ase(mlip_bin=self.hyperparameters['mlip_bin'], 
                                        atoms=atoms, 
                                        mpirun=bin_pref, 
                                        pot_path=self.trained_pot['path'], 
                                        cfg_files=False, 
                                        dir=wdir,
                                        write_conf=False)
        return atoms_calc
            
        
    def save_model(self, filepath):
        '''We want to store this model as JSON'''
        new = dc(self)
        # We need to transform the dataset into text (.cfg format)
        
        new.train_set = conv_ase_to_mlip2_text(self.train_set, props=True)
        if self.ml_trainset is not None:
            new.ml_trainset = conv_ase_to_mlip2_text(self.ml_trainset, props=True)
        with open(filepath, 'w') as fl:
            json.dump(new.__dict__, fl, indent=4)

    def upload_potfile(self, potfile_path, level, dataset=None, regular=True, adapt_hyperpars=False):
        print(f'Uploading the potfile {potfile_path}.')
        pars_to_adapt = []
        min_dist, max_dist, species_count, radial_basis_size, radial_basis_type = extract_mtp_info_from_pot(potfile_path)

        if min_dist != self.min_dist:
            pars_to_adapt.append('min_dist')
            if adapt_hyperpars:
                self.min_dist = min_dist

        if max_dist != self.max_dist:
            pars_to_adapt.append('max_dist')
            if adapt_hyperpars:
                self.max_dist = max_dist

        if level != self.mtp_level:
            pars_to_adapt.append('mtp_level')
            if adapt_hyperpars:
                self.mtp_level = level

        if species_count != self.species_count:
            pars_to_adapt.append('species_count') # is no dataset is given and self.train_set exists, species_count could be different between this pot and the self.train_set
                                                  # for this reason, a few lines below, the model is irregularised
            if adapt_hyperpars:
                self.species_count = species_count

        if radial_basis_size != self.radial_basis_size:
            pars_to_adapt.append('radial_basis_size')
            if adapt_hyperpars:
                self.radial_basis_size = radial_basis_size
        if radial_basis_type != self.radial_basis_type:
            pars_to_adapt.append('radial_basis_type')
            if adapt_hyperpars:
                self.radial_basis_type = radial_basis_type

        if len(pars_to_adapt) > 0:
            print(f'The following hyperparameters are different from the preexisting ones (or do exist now but didn\'t exist before):')
            print(pars_to_adapt)
            if adapt_hyperpars:
                print('They were overwritten (or just written in case they didn\'t exist before).')
            else:
                print('The upload cannot be done; please set adapt_hyperpars = True if you wish to overwrite.')
                return
            
        self.trained_pot_path = potfile_path
        with open(potfile_path, 'r') as fl:
            self.trained_pot = fl.readlines()
        if dataset is not None:
            if self.train_set is not None:
                print('A new dataset was given, so it will overwrite the preexisting one.')
            self.train_set = dataset 
            if regular:
                self.make_regular()
        else:
            if self.train_set is not None:
                self.make_irregular()
    



           


    
    
    
    
    @classmethod
    def MTP_model_from_potfile(cls, potfile, level, dataset=None):
        min_dist, max_dist, species_count, radial_basis_size, radial_basis_type = extract_mtp_info_from_pot(potfile)
        instance = cls(level=level,
                       min_dist=min_dist,
                       max_dist=max_dist,
                       radial_basis_size=radial_basis_size,
                       radial_basis_type=radial_basis_type,
                       train_set=dataset)     
        instance.is_trained = True
        instance.species_count = species_count
        return instance
        
        