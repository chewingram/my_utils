import json
from copy import deepcopy as dc
from ..utils_mlip import train_pot_from_ase_tmp as train_pot_from_ase, conv_ase_to_mlip2_text
from ..utils import flatten



class MTP_model():
    def __init__(self,
                 mtp_level,
                 min_dist,
                 max_dist,
                 radial_basis_size,
                 radial_basis_type,
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
        self.mtp_level = mtp_level
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.radial_basis_size = radial_basis_size
        self.radial_basis_type = radial_basis_type
        self.train_set = train_set
        self.ml_trainset = None # we need to initialize it
        self.trainset_bck = None # we need to initialize it
        if self.train_set is not None:
            self.species_count = set(flatten([x.get_chemical_symbols for x in self.train_set]))
        else:
            self.species_count = None
        self.is_trained = False
        self.is_regular = True # since no trainset is there, we can consider it regular, even if not trained
    
    def train(self, 
              mlip_bin,
              untrained_pot_file_dir,
              training_dir,
              bin_pref,
              final_evaluation=True,
              train_set=None,
              params=None):
        '''
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
        if train_set is None:
            txt = ''
            if self.train_set is not None:
                train_set = self.train_set
            else:
                raise ValueError('To train the potential either the MTP_model object must have the train_set or the parameter train_set must be passed to this method.')
        else:
            txt1 = '; the preexisting trainset has been overwritten with the new dataset provided.'
            txt2 = '; therefore, the preexisting trainset was preserved (not replaced by the new dataset provided)'
        results = train_pot_from_ase(mlip_bin=mlip_bin,
                                    untrained_pot_file_dir=untrained_pot_file_dir,
                                    mtp_level=self.mtp_level,
                                    min_dist=self.min_dist,
                                    max_dist=self.max_dist,
                                    radial_basis_size=self.radial_basis_size,
                                    radial_basis_type=self.radial_basis_type,
                                    train_set=train_set,
                                    dir=training_dir,
                                    params=params,
                                    mpirun=bin_pref,
                                    final_evaluation=final_evaluation)
        
        # we need to check that the training went to completion
        with open(training_dir,joinpath('log_train'), 'r') as fl:
            lines = fl.readlines()
            if any(['* * * TRAIN ERRORS * * *' in line for line in lines]):
                print('Training done successfully' + txt1)
                self.train_set = train_set # in case train_set was given, we want to set it as self.train_set only if the training was successful
            else:
                print('The training was not successful' + txt2)
                return

        self.species_count = len(set(flatten([x.get_chemical_symbols() for x in train_set])))

        if final_evaluation == True:
            trained_pot_file_path = results[0]
            ml_trainset = results[1]
            errs = results[2]
            self.ml_trainset = ml_trainset
        else:
            trained_pot_file_path = results
            
        self.trained_pot_path = trained_pot_file_path
        with open(trained_pot_file_path, 'r') as fl:
            self.trained_pot = fl.readlines()
        
        self.is_trained = True
        self.make_regular()
    
    def make_regular(self):
        '''If there is trainset_bck, it will be removed'''
        if self.trainset_bck not None:
            self.trainset_bck = None
        self.is_regular = True
    
    def make_irregular(self):
        self.is_regular = False
    
    def set_trainset(self, dataset):
        if self.is_regular:
            self.make_irregular()
        self.trainset_bck = self.train_set
        self.train_set = dataset

    def unset_trainset(self):
        '''If there is trainset_bck, it will remain'''
        self.train_set = None
        self.is_regular = True
    
    def reset_trainset_bck(self):
        '''If there is self.trainset_bck, it is assigned to self.train_set and then removed'''
        print('Reset of the previous train set backup')
        if self.trainset_bck is not None:
            self.train_set = self.trainset_bck
            self.trainset_bck = None
        else:
            print('No preexisting train set backup; aborted!')
        
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
        
        