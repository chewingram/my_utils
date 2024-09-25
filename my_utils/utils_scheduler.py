import os
import sys
import pickle as pkl
import shutil
import subprocess
import warnings

def warn(text):
    warnings.warn(text)

def mkdir_warner(dir_path, interactive=False):
    '''Create a directory. If it already exists, the old one will be deleted. If interactive=True, the user will be asked whether to delete or not

    Parameters:
    dir_path (str): path of the directory to create
    interactive (bool): True = if the directory already exists the user will be asked whether to delete it and overwrite or stop. False = if the directory already exists it will be automatically deleted and overwritten without asking the user.
    '''
    if not dir_path[-1] == "/":
        dir_path = dir_path + "/"

    if os.path.exists(dir_path):
        if interactive == True:
            chk = input("The directory \"" + dir_path +"\" already exists. It will be deleted, if you're ok with it please type \"y\".")
            if not chk == "y":
                print("You chose not to delete the folder, so the program will stop. Bye.")
                sys.exit()
        shutil.rmtree(dir_path, ignore_errors=True)
    os.mkdir(dir_path)
    
def write_file(text, filepath):
    with open(filepath, 'w') as f:
        f.write(text)
        
def save_pkl(obj, filename=None):
    if filename is None:
        filename = "object.pkl"
    elif not isinstance(filename, str):
            raise TypeError("The file name must be a string.")

    with open(filename, "wb") as f:
        pkl.dump(obj, f)

def load_pkl(filename=None):
    if filename is None:
        filename = "object.pkl"
    elif not isinstance(filename, str):
        raise TypeError("A string must be provided as path.")
    elif not os.path.exists(filename):
        raise FileNotFoundError(f"Couldn't find {filename}.")
    with open(filename, 'rb') as f:
        return pkl.load(f)
    
def dict_ext(dictionary, exclude=None):
    '''
    This function takes a dictionary as input and returns a list whose each elements is a list of one key-value couple. 
    '''
    if not isinstance(dictionary, dict):
        raise TypeError("The input parameter must be a dictionary.")
    if exclude is None:
        exclude == []
    if not isinstance (exclude, list):
        exclude = [exclude]
    dict_key = [x for x in dictionary]
    dict_val = [dictionary[x] for x in dict_key]
    z = [[x,y] for x, y in zip(dict_key, dict_val) if x not in exclude]
    return z

def slurm_in_queue(slurm_id):
    out = subprocess.run(['squeue'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    out = out.split('\n')
    for x in out:
        for y in x.split():
            if str(slurm_id) in y:
                return True
    return False
            