from abc import ABC, abstractmethod
from pathlib import Path
import traceback

from ase.atoms import Atoms


class MlipModel(ABC):

    def __init__(self, root_dir, train_set=None, hyperparameters=None):
        
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(exsist_ok=True, parents=True)
        self.training_dir = root_dir.joinpath('training')
        self.mkdir(exsist_ok=True, parents=True)

        if hyperparameters is not None:
            self.hyperparameters = hyperparameters
        else:
            self.hyperparameters = dict()

        self.train_set = train_set
        self.ml_trainset = None # we need to initialize it
        self.trainset_bck = None # we need to initialize it
        if self.train_set is not None:
            self.species_count = set(flatten([x.get_chemical_symbols for x in self.train_set]))
        else:
            self.species_count = None
        
        self.is_trained = False
        self.training_dir = None 
        self.regular = dict(d=False, h=False) # d = dataset, h = hyperparameters


    
    def is_regular(level=''):
        if level == '':
            if self.regular['d'] == True and self.regular['h'] == True:
                return True
            else:
                return False
        elif level == 'd':
            if self.regular['d'] == True:
                return True
            else:
                return False
        elif level == 'h':
            if self.regular['h'] == True:
                return True
            else:
                return False
            
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def store_trained(self):
        pass

    def train_model(self, train_set=None, training_hyperparameters=None, bin_pref='', final_evaluation):
        '''
        Three possible models can be there:
        1. (all) regular
        2. trained but irregular (both partially or totally)
        3. untrained (with or without dataset and hyperpars)
        In any case, if dataset or hyperpars are passed, a backup must be done and resume/deleted if the training is
        failed/successful. If the training is failed, we must restore the model to it's status before entering in the 
        training function.
        '''
        # save the status
        train_set_bck = self.train_set,
        hyperparameters_bck = self.hyperparameters
        
        if train_set is None:
            if self.train_set is None:
                raise ValueError('To train the potential, either the model object must have a training set, or the parameter `train_set` must be passed to this method.')
        else:
            if self.train_set is not None:
                txt1 = '; the preexisting trainset has been overwritten with the new dataset provided.'
                txt2 = '; therefore, the preexisting trainset was preserved (not replaced by the new dataset provided)'
            else:
                txt1 = ''
                txt2 = ''
            self.train_set = train_set

        if training_hyperparameters is not None:
            self.hyperparameters = training_hyperparameters
       
        # run the training
        try:
            success = self.train(bin_pref=bin_pref)
        except:
            success = False
            # we need to restore the old things
            self.train_set = train_set_bck
            self.hyperparameters = hyperparameters_bck
            print('The training was not successful' + txt2)
            print('Please, find the traceback below.')
            raise

        # from here only if sucess is True
        
        print('Training done successfully' + txt1)
        self.is_trained = True
        del train_set_bck
        del hyperparameters_bck
        self.make_regular()
        return success
            

    def _make_irregular(self, level=''):
        if level == '':
            self.regular['d'] = True
            self.regular['h'] = True
        elif level == 'd':
            self.regular['d'] = True
        elif level == 'h':
            self.regular['h'] = True
    
    def _make_irregular(self, level=''):
        if level == '':
            self.regular['d'] = False
            self.regular['h'] = False
        elif level == 'd':
            self.regular['d'] = False
        elif level == 'h':
            self.regular['h'] = False
    
    def is_regular(self, level=''):
        if level=='':
            if self.regular['h'] == True and self.regular['d'] == True:
                return True
            else:
                return False
        elif level == 'd':
            if self.regular['d'] == True:
                return True
            else:
                return False
        elif level == 'h':
            if self.regular['h'] == True:
                return True
            else:
                return False
            
    
    def compute_properties(self, atoms, wdir, parameters):
        if not self.is_trained:
            raise ValueError('The calculation cannot be done because the model is not trained!')
        elif not self.is_regular():
            raise ValueError('The calculation cannot be done because the model is not regular!')
        
        atoms_calc = self._compute_properties(atoms=atoms, wdir=wdir, **parameters)
        return atoms_calc
    

    @abstractmethod
    def _compute_properties(self, atoms, wdir, **kwargs):
    pass    

    
    
    # def _set_trainset(self, dataset):
    #     if (not self.is_trained and self.train_set is not None) or self.is_regular('d'):
    #         self.trainset_bck = self.train_set.copy()
    #     self.train_set = dataset
    #     self._make_irregular('d')
    
    # def _set_hyperparameters(self, hyperparameters):
    #     if (not self.is_trained and self.hyperparameters is not None) or self.is_regular('h'):
    #         self.hyperparameters_bck = self.hyperparameters.copy()
    #     self.hyperparameters = hyperparameters
    #     self._make_irregular('h')
    
    # def _delete_backups(self, level=''):
    #     if level == '':
    #         self.hyperparameters_bck = None
    #         self.trainset_bck = None
    #     elif level == 'd':
    #         self.trainset_bck = None
    #     elif level == 'h':
    #         self.hyperparameters_bck = None


    def _unset_trainset(self):
        '''If there is trainset_bck, it will remain'''
        self.train_set = None
        self.is_regular = True
    
    # def _reset_trainset_bck(self):
    #     '''If there is self.trainset_bck, it is assigned to self.train_set and then removed'''
    #     print('Reset of the previous train set backup')
    #     if self.trainset_bck is not None:
    #         self.train_set = self.trainset_bck
    #         self.trainset_bck = None
    #     else:
    #         print('No preexisting train set backup; aborted!')

    # def _reset_hyperparameters_bck(self):
    #     '''If there is self.hyperparameters_bck, it is assigned to self.hyperparameters and then removed'''
    #     print('Reset of the previous hyperparameters backup')
    #     if self.hyperparameters_bck is not None:
    #         self.hyperparameters = self.hyperparameters_bck.copy()
    #         self.hyperparameters_bck = None
    #     else:
    #         print('No preexisting hyperparameters backup; aborted!')