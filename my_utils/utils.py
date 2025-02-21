import warnings
import numpy as np
import matplotlib
import os 
import sys
import shutil 
import numbers
from pathlib import Path
import logging
from scipy.stats import linregress
from ase.io import read, write


def mkdir_warner(dir_path, interactive=False):
    '''Create a directory. If it already exists, the old one will be deleted. If interactive=True, the user will be asked whether to delete or not 
    
    Parameters:
    dir_path (str): path of the directory to create
    interactive (bool): True = if the directory already exists the user will be asked whether to delete it and overwrite or stop. False = if the directory already exists it will be automatically deleted and overwritten without asking to the user.
    '''
    
    import os
    import sys
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

def data_reader(filename, separator, n_lines=-1, skip_lines=0, frmt="str", red_space=True):
    '''Read data from a text organized in columns

    Parameters:
    filename (str): path to the file to read (incl. name)
    separator (str): string to look for as separator between values belonging to different columns. If it's set to "", then the string will be tokenized as by the function split() with no arguments
    n_lines (int): number of lines to read. If n_lines=-1 (default), then all the lines are read.
    skip_lines (int): number of lines to skip from the beginning
    frmt (str): format to convert the data into. It can be "str" or "float"
    red_space (bool): True = if the separator does not exist in the line, it will look for the separator without the last character
    Returns:
    list:data, each element of data is a list of n elements, each of which is an element of the line (for n lines)

   '''
    
    data = []
    if separator != "":
        separator2 = separator[:-1]
    with open(filename) as fl:
        data = [line.rstrip() for line in fl]
    # Now data is a list of string
    # Let's convert each string into a list of elements (token separated by space)
    if n_lines == -1:
        n_lines = len(data)

    for i in range(n_lines):
        if separator == "":
            data[i] = data[i].split()
        elif separator not in data[i]:
            data[i] = data[i].split(separator2)
        else:
            data[i] = data[i].split(separator)

            
    # Remove the first rm_lines lines
    for i in range(skip_lines):
        data.pop(0)
    # Convert the data into the format
    if frmt == "float":
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = float(data[i][j])
    elif frmt == "str":
        pass
    else:
        print("Format not valid!")
        exit()
    
    return data


def data_writer(filename, data, header_pieces=None, header_separator=" ", overwrite=True):
    '''Write data contained in a list of listes. Each element of the list has to be a list and each element of it will be printed in a line separated by "header_separator"
    
    Parameters:
    filename (str): path to the file to write
    data (list): each element of the list has to be a list and each element of it will be printed in a line separated by "header_separator"
    header_pieces (list): each element of this list will be printed after a "# " as header, separated by "header_separator"
    header_separator (str): string to separate the data with
    overwrite (bool): if it's set to "True", then if the file already exists it will be overwritten, otherwise it will be appended. 
    '''
    import os
    import sys
    
    original_stdout = sys.stdout
    if header_pieces == None:
        header_pieces = []   
    if os.path.exists(filename) and overwrite == True:
        os.remove(filename)    
    with open(filename, "a") as fl: 
        sys.stdout = fl
        if len(header_pieces) > 0: 
            header = "# " + header_pieces
            print(header)
        for i in range(len(data)):
            line = str(data[i][0])
            for j in range(1, len(data[i])):
                line = line + header_separator + str(data[i][j])
            # Print the line
            print(line)
    sys.stdout = original_stdout


def dir_cleaner(path):
    '''Delete the content of a directory (except for read-only files)
    Parameters:
    path (str): path of the directory to clean
    '''
    
    import os
    list_files = os.listdir(".")
    for x in list_files:
        os.remove(x)
    
def repeat(elem, n):
    '''
    Function to create a list of repeated INDEPENDENT elements. 
    '''
    lst = []
    if isinstance(elem, list):
        for i in range(n):
            lst.append(elem.copy())
    else:
        for i in range(n):
            lst.append(elem)
    
    return lst

def warn(text):
    import warnings
    '''
    Function to raise a warning without too much text around
    '''
    warnings.warn(text)
    
def flatten(l):
    """Recursively flattens a nested list."""
    if not isinstance(l, list):
        raise TypeError("The object given is not a list!")
    
    res = []
    for x in l:
        if isinstance(x, list):
            res.extend(flatten(x))  # Always recursively flatten lists
        else:
            res.append(x)  # Append non-list elements directly
    
    return res

def vec_diff_eval(vec1, vec2, rad_tol, ang_tol, zero_diff=0.001, dot_round=5):
    '''Evaluate whether two numpy vectors are equal or not within a certain tolerance
    Parameters:
    vec1 (np.array): vector n. 1.
    vec2 (np.array): vector n. 2.
    rad_tol (float): threshold of the difference in length between the two vectors (normalized by the minimum lenght)
                     beyond which the two vectors are considered different, independent on the angular part.
    ang_tol (float): threshold of the angle between the two vectors beyond which they are considered different,
                     independent on the radial part.
    zero_diff (float): threshold of the difference in lenght between two vectors (one of which is exactly zero),
                       beyond which the two vectors are considered different
    dot_round (int): number of digits to round the dot product between the two vectors. Without rounding, number 
                     slightly bigger than 1 (or less then -1) can be obtained, such as 1.0000002, and np.arccos()
                     can't deal with them.
    '''
    # Evaluate the difference between two numpy vectors
    import numpy as np
    
    min_length = min(np.linalg.norm(vec1), np.linalg.norm(vec2))
    if min_length == 0:
        if (abs(np.linalg.norm(vec1)) <= zero_diff):
            return True
        else:
            return False
    else:
        dot_prod = np.dot(vec1/np.linalg.norm(vec1), vec2/np.linalg.norm(vec2))
        if ((abs(np.linalg.norm(vec1) - np.linalg.norm(vec2)) / min_length) <= rad_tol) and \
             (np.arccos(round(dot_prod, dot_round))) <= ang_tol:
            return True
        else:
            return False
        
def norm_np(vec):
    import numpy as np
    return vec / np.linalg.norm(vec)

def cap_first(string):
    '''
    Capitalize the first character of a string
    Arguments:
    string(str): string
    '''
    string = string[0].capitalize() + string[1:]
    return string

def low_first(string):
    '''
    Lower case the first character of a string
    Arguments:
    string(str): string
    '''
    string = string[0].lower() + string[1:]
    return string

def space(n):
    '''
    Return n spaces
    Arguments:
    n(int): number of spaces
    '''
    t = ''
    for i in range(n):
        t += ' '
    return t

def inv_dict(dictionary):
    '''
    Take a dictionary as input and return its "inverse" map. The values become the keys and viceversa.
    !! Beware! The new keys are the string representation of the values they were in the original dictionary. !!
    '''
    s = dict()
    for x in dictionary.keys():
        s[str(dictionary[x])] = x
    return s  

def mae(data1, data2):
    '''
    Compute the mae between two parallel sets of numbers.
    Arguments:
    data1(list, numpy.array): set 1
    data2(list, numpy.array): set 2
    '''
    if isinstance(data1, numbers.Number):
        data1 = np.array([data1])
    if not isinstance(data1, list) and not isinstance(data1, type(np.array([42]))):
        raise TypeError('data1 must be a list or a numpy array!')
    elif not all([isinstance(x, numbers.Number) for x in data1]):
        raise TypeError('All elements of data1 must be numbers!')
    if isinstance(data1, list):
        data1 = np.array(data1)
    
    if isinstance(data2, numbers.Number):
        data2 = np.array([data2])
    if not isinstance(data2, list) and not isinstance(data2, type(np.array([42]))):
        raise TypeError('data2 must be a list or a numpy array!')
    elif not all([isinstance(x, numbers.Number) for x in data2]):
        raise TypeError('All elements of data1 must be numbers!')
    if isinstance(data2, list):
        data2 = np.array(data2)        
    
    if len(data1) != len(data2):
        raise ValueError('The two array must have the same size!')
    
    ae = np.absolute(data1 - data2)
    mae = np.mean(ae)
    #print(f'data1: {data1}, data2: {data2}, ae: {ae}, mae: {mae}')
    return mae

def rmse(data1, data2):
    '''
    Compute the rmse between two parallel sets of numbers.
    Arguments:
    data1(list, numpy.array): set 1
    data2(list, numpy.array): set 2
    '''
    if isinstance(data1, numbers.Number):
        data1 = np.array([data1])
    if not isinstance(data1, list) and not isinstance(data1, type(np.array([42]))):
        raise TypeError('data1 must be a list or a numpy array!')
    elif not all([isinstance(x, numbers.Number) for x in data1]):
        raise TypeError('All elements of data1 must be numbers!')
    if isinstance(data1, list):
        data1 = np.array(data1)
    
    if isinstance(data2, numbers.Number):
        data2 = np.array([data2])
    if not isinstance(data2, list) and not isinstance(data2, type(np.array([42]))):
        raise TypeError('data2 must be a list or a numpy array!')
    elif not all([isinstance(x, numbers.Number) for x in data2]):
        raise TypeError('All elements of data1 must be numbers!')
    if isinstance(data2, list):
        data2 = np.array(data2)        
    
    if len(data1) != len(data2):
        raise ValueError('The two array must have the same size!')
    
    se = (data1 - data2) ** 2
    mse = np.mean(se)
    rmse = mse ** 0.5
    #print(f'data1: {data1}, data2: {data2}, se: {se}, mse: {mse}, rmse: {rmse}')
    return rmse

def R2(data1, data2):
    '''
    Compute the R2 between true and observed/predicted data.
    Arguments:
    data1(list, numpy.array): true data
    data2(list, numpy.array): observed/predicted data
    '''
    if isinstance(data1, numbers.Number):
        data1 = np.array([data1])
    if not isinstance(data1, list) and not isinstance(data1, type(np.array([42]))):
        raise TypeError('data1 must be a list or a numpy array!')
    elif not all([isinstance(x, numbers.Number) for x in data1]):
        raise TypeError('All elements of data1 must be numbers!')
    if isinstance(data1, list):
        data1 = np.array(data1)
    
    if isinstance(data2, numbers.Number):
        data2 = np.array([data2])
    if not isinstance(data2, list) and not isinstance(data2, type(np.array([42]))):
        raise TypeError('data2 must be a list or a numpy array!')
    elif not all([isinstance(x, numbers.Number) for x in data2]):
        raise TypeError('All elements of data1 must be numbers!')
    if isinstance(data2, list):
        data2 = np.array(data2)        
    
    if len(data1) != len(data2):
        raise ValueError('The two array must have the same size!')
    
    mean = np.mean(data1)
    rss = np.sum((data2 - data1)**2)# residual sum of squares
    tss = np.sum((data2 - mean)**2) # total sum of squares
    R2 = 1 - rss / tss
    #print(f'data1: {data1}, data2: {data2}, rss: {rss}, tss: {tss}, R2: {R2}')

    slope, intercept, r_value, p_value, std_err = linregress(data1, data2)
    return r_value**2

def time_to_three(time, unit='s'):
    import math 
    if unit == 'h':
        h = math.floor(time)
        m = math.floor((time - h) * 60)
        s = int(round((time - h - m/60) * 60 * 60, 0))
    if unit == 'm':
        h = math.floor(time/60)
        m = math.floor((time - h * 60))
        s = int(round((time - h * 60 - m) * 60, 0))
    if unit == 's':
        h = math.floor(time/60/60)
        m = math.floor((time - h * 60 * 60) / 60)
        s = int(round((time - h * 60 * 60 - m *60), 0))
    return f'{h:0>2}:{m:0>2}:{s:0>2}'

def three_to_time(three, unit='s'):
    tmp = three.split(':')
    num = [int(x) for x in tmp]
    sh = num[2] / 60 / 60
    mh = num[1] / 60
    tmp = num[0] + mh + sh

    three = time_to_three(tmp, 'h')
    # now three is in a good format
    tmp = three.split(':')
    num = [int(x) for x in tmp]

    h = num[0] * 3600# in s unit
    m = num[1] * 60
    s = num[2]
    time = h + m + s
    if unit == 'h':
        conv = 1 / 3600
    elif unit == 'm':
        conv = 1 / 60
    elif unit == 's':
        conv = 1
    return time * conv

def from_list_of_numbs_to_text(list):
    text = ''
    for line in list:
        for j, elem in enumerate(line):
            if j == len(line) - 1:
                text += f'{elem:1.15f}'
            else:
                text += f'{elem:1.15f}\t'
        text += '\n'
    return text

def from_list_to_text(list):
    txt = ''
    for i, line in enumerate(list):
        txt += f'{line}'
    return txt

def logbase(base, argument):
    return np.log(argument)/np.log(base)


def path(path):
    path = os.path.abspath(path)
    if not path.endswith('/'):
        path = path + '/'
    return path

def ln_s_f(origin, dest):
    path1 = Path(origin)
    path2 = Path(dest)
    print(path1, path2)
    if path2.is_dir():
        os.system(f'ln -s -f {path1.absolute()} {path2.joinpath(path1.name).absolute()}')
    else:
        os.system(f'ln -s -f {path1.absolute()} {path2.absolute()}')
        
        


# def setup_logging(tool_name, debug=False,log_file="logger.log"):
#     """
#     Set up a logger and exception hook for a tool.

#     Parameters:
#         tool_name (str): Name of the tool to identify log messages.
#         debug (bool): activate debug logging
#         log_file (str): File where logs will be saved.
#         log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

#     Returns:
#         logger (logging.Logger): Configured logger.
#         handler (logging.Handler): Console handler for additional customization.
#     """
    
#     # Create a logger with the tool's name
#     logger = logging.getLogger(tool_name)
#     logger.setLevel(logging.DEBUG)  # Allow all levels; filter in handlers

#     if debug == True:
#         log_level = logging.DEBUG
#     else:
#         log_level = logging.INFO
        
#     # Create a file handler for writing logs
#     file_handler = logging.FileHandler(log_file)
#     file_handler.setLevel(log_level)  # Default to INFO for file
#     file_formatter = logging.Formatter('%(message)s')
#     file_handler.setFormatter(file_formatter)

#     # Create a console handler for interactive output
#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(log_level)  # Default to INFO for console
#     console_formatter = logging.Formatter('%(message)s')
#     console_handler.setFormatter(console_formatter)

#     # Add handlers to the logger
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

#     # Define a custom exception hook for uncaught exceptions
#     def handle_exception(exc_type, exc_value, exc_traceback):
#         if issubclass(exc_type, KeyboardInterrupt):
#             # Use default behavior for KeyboardInterrupt
#             sys.__excepthook__(exc_type, exc_value, exc_traceback)
#             return
#         logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

#     # Set the custom exception hook
#     sys.excepthook = handle_exception

#     return logger, console_handler
    
def setup_logging(logger_name=None, log_file=None, debug=False):
    """
    Set up a logger and exception hook for a tool.

    Parameters:
    -----------
        logger_name: str
            Name of the existing logger (if any) to identify log messages. If None, use the root logger.
        log_file: str
            File where logs will be saved. If None, no file handler is added.
        debug: bool
            activate the debug logging

    Returns:
        logger: logging.Logger
            Configured logger.
            
    """
    
    # If tool_name is None, use the root logger
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logging.getLogger()
        logging.propagate = False

    # Avoid duplicate handlers if the logger already has handlers
    if not logger.handlers:
        # Set logger level
        logger.setLevel(logging.DEBUG)
        
        log_level = logging.DEBUG if debug == True else logging.INFO
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (if log_file is provided)
        if log_file:
            if Path(log_file).is_file():
                Path(log_file).unlink()
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Define a custom exception hook
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            print('error to transmit')
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = handle_exception

    return logger


def atlen(*argv):
    if len(sys.argv) < 2:
        fname = 'Trajectory.traj'
    else:
        fname = sys.argv[1]
    at = read(fname, index=':')
    l = len(at)
    print(f'The file {fname} contains {l} configurations.')
    
def mute_logger(name="mute_logger"):
    """
    Create a logger that discards all log messages.

    Parameters:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A logger that doesn't output any messages.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to the lowest level to allow all messages
    logger.addHandler(logging.NullHandler())  # Attach a NullHandler to discard messages
    return logger   
        

def min_distance_to_surface(cell):
    """
    Compute the minimum distance from the center to the surface of a parallelepiped
    given a 3x3 cell matrix (each row is a lattice vector).
    
    Parameters:
        cell (numpy.ndarray): 3x3 matrix where rows are the three lattice vectors.
    
    Returns:
        float: The minimum distance from the center to the surface.
    """
    # Extract vectors from the cell matrix (each row corresponds to a lattice vector)
    v1, v2, v3 = cell[0], cell[1], cell[2]

    # Compute face distances (perpendicular distances to three non-equivalent faces)
    d_face1 = 0.5 * np.linalg.norm(np.cross(v2, v3)) / np.linalg.norm(v1)
    d_face2 = 0.5 * np.linalg.norm(np.cross(v3, v1)) / np.linalg.norm(v2)
    d_face3 = 0.5 * np.linalg.norm(np.cross(v1, v2)) / np.linalg.norm(v3)

    # Compute edge distances (shortest distances to edges)
    d_edge12 = 0.5 * np.linalg.norm(np.cross(v1, v2)) / np.sqrt(np.linalg.norm(v1)**2 + np.linalg.norm(v2)**2)
    d_edge23 = 0.5 * np.linalg.norm(np.cross(v2, v3)) / np.sqrt(np.linalg.norm(v2)**2 + np.linalg.norm(v3)**2)
    d_edge31 = 0.5 * np.linalg.norm(np.cross(v3, v1)) / np.sqrt(np.linalg.norm(v3)**2 + np.linalg.norm(v1)**2)

    # Compute vertex distance (shortest distance to the nearest vertex)
    d_vertex = 0.5 * np.linalg.norm(v1 + v2 + v3) / np.sqrt(3)

    # Find the minimum distance among face, edge, and vertex distances
    return min(d_face1, d_face2, d_face3, d_edge12, d_edge23, d_edge31, d_vertex)

def mic_sign(vec):
    '''
    Miminum image convention applying pbc preserving the sign
    vec must be in reduced coordinates, np.arrays
    '''
    return vec - np.round(vec)


def lc(l):
    '''
    Transforms a string (or strings of list recursively, if l is a list) into lower-case and capitalized strings
    e.g. lc([['lo', 'oO'], 'ciao', ['pp']]) -----> [['Lo', 'Oo'], 'Ciao', ['Pp']]
    '''
    if isinstance(l, list):
        ll = []
        for item in l:
            ll.append(lc(item))    
    else:
        ll = l.lower().capitalize()
    return ll

def at_extract(*argv):

    def check_number(x):
        if x[0] == "-":
            x = x[1:]
        if x.isdigit():
            return True
        else:
            return False

    trajname = sys.argv[1]
    indices = sys.argv[2]
    if len(sys.argv) == 4:      
        savename = sys.argv[3]
    else:
        savename = 'extr_' + trajname
    ats = read(trajname, index=':')
    if ":" in indices:
        indices = indices.split(":")
    else:
        indices = [indices]
    if not all([check_number(x) for x in indices]):
        raise ValueError('Please use a valid index or slice of indices')
    else:
        indices = [int(x) for x in indices]
    
    if len(indices) == 1:
        extr = [ats[indices[0]]]
    elif len(indices) == 2:
        extr = ats[indices[0]:indices[1]]
    elif len(indices) == 3:
        extr = ats[indices[0]:indices[1]:indices[2]]
    else:
        raise ValueError('Please use a valid index or slice of indices')
    
    write(savename, extr)
