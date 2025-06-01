from ..utils_mlip import train_pot_from_ase_tmp as train_pot_from_ase
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

        self.is_trained = False
    
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
        if train_set is not None:
            self.train_set = train_set
        elif self.train_set is None:
            raise ValueError('To train the potential either the MTP_model object must have the train_set or the parameter train_set must be passed to this method.')

        results = train_pot_from_ase(mlip_bin=mlip_bin,
                                    untrained_pot_file_dir=untrained_pot_file_dir,
                                    mtp_level=self.mtp_level,
                                    min_dist=self.min_dist,
                                    max_dist=self.max_dist,
                                    radial_basis_size=self.radial_basis_size,
                                    radial_basis_type=self.radial_basis_type,
                                    train_set=self.train_set,
                                    dir=training_dir,
                                    params=params,
                                    mpirun=bin_pref,
                                    final_evaluation=final_evaluation)
        
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
        
        