import os
import numpy as np
from pathlib import Path
import abc

from ase.atoms import Atoms
from ase.data import atomic_numbers
from ase.io import read, write
from ase.build import make_supercell
from ase.io.abinit import read_abinit_out
from ase.calculators.singlepoint import SinglePointCalculator

def extract_structure_for_abinit_input(structure, pseudos_dir, pseudos_names=None, pseudos_extension=None):
    '''
    Return a list of lines with acell, rprim, ntypat, znucl, typat, xred
    '''
    lines = []
    cell = structure.get_cell()
    atom_numbers = structure.get_atomic_numbers()
    unique_numbers = [str(x) for x in list(set(atom_numbers))]
    # we need to sort unique_numbers in alphabetic order of the relative chemical symbols
    inv_atomic_numbers = {str(y): str(x) for x, y in atomic_numbers.items()}
    unique_numbers = sorted(unique_numbers, key=lambda num: inv_atomic_numbers[num])

    ntypat = len(unique_numbers)
    typats = dict(zip( unique_numbers, list(range(1, ntypat+1)) ))
    typat = ' '.join([f'{typats[str(x)]}' for x in atom_numbers])
    atomic_positions = structure.get_scaled_positions()
    lines = []
        # structure
    lines.append(f'acell 1.0 1.0 1.0 Angstrom\n')
    lines.append(f'\n')
    lines.append(f'rprim\n')
    lines.append(f'{cell[0,0]} {cell[0,1]} {cell[0,2]}\n')
    lines.append(f'{cell[1,0]} {cell[1,1]} {cell[1,2]}\n')
    lines.append(f'{cell[2,0]} {cell[2,1]} {cell[2,2]}\n')
    lines.append(f'\n')
    lines.append(f'natom {len(atom_numbers)}\n')
    lines.append(f'ntypat {ntypat}\n')
    lines.append(f'\n')
    lines.append(f'znucl {" ".join(unique_numbers)}\n')
    lines.append(f'\n')
    lines.append(f'typat\n')
    lines.append(f'{typat}\n')
    lines.append(f'\n')
    lines.append(f'xred\n')
    for position in atomic_positions:
        lines.append(f'{position[0]:.20f} {position[1]:.20f} {position[2]:.20f}\n')
    if pseudos_names is not None:
        pseudos_names.sort()
        pseudos_names = [f'{x}.{pseudos_extension}' for x in pseudos_names]
    else:
        pseudos_names = []
        inv_typats = {str(v): k for k, v in typats.items()}
        for i in range(len(typats)):
            numb = inv_typats[str(i+1)]
            #symb = list(set(structure.get_chemical_symbols()))[i].lower()
            symb = inv_atomic_numbers[str(numb)].lower()
            name = f"{numb}{symb}.{pseudos_extension}"
            pseudos_names.append(name)
    lines.append(f"pseudos \"{', '.join(pseudos_names)}\"\n")
    lines.append(f"pp_dirpath \"{Path(pseudos_dir)}\"\n")
    return lines


def create_abinit_input(input_params=dict(), pseudos_dir='./', pseudos_format='custom', pseudos_extension='', pseudos_names=None, structure=None, save=False, filepath=None):
            '''
            Function to create an ABINIT input.
            Args:
            input_params(dict): dictionary with the abinit variables and values; only numbers or strings
            pseudos_dir(str): directory where the pseudos are
            pseudos_format(str): 
                                - custom: a list of pseudos names, excluding the extension, must be given
                                - ASE-MLACS: the list of names will be generated assuming they are of the form 1h, 2he, 3li, 16s
            pseudos_extension(str): extension of the pseudos
            pseudos_names(list): only for pseudos_format='custom'; list with the names (EXCLUDING THE EXTENSION) of the pseudos
            structure(ase.atoms.Atoms): ase Atoms object with the structure
            save(bool): True: the input file will be written
            filepath(str): mandatory if save=True (ignored otherwise); path of the file to write the input 
            '''
            if pseudos_format == 'custom':
                if pseudos_names is None:
                    raise ValueError('When pseudos_format = custom, pseudos_name must be a list with the names (EXTENSION EXCLUDED) of the pseudos')
            else:
                pseudos_names = None
            lines = ["# Input for ABINIT written by my_utils #\n"]
                # structure
            lines.extend(extract_structure_for_abinit_input(structure, pseudos_dir=pseudos_dir, pseudos_extension=pseudos_extension, pseudos_names=pseudos_names))
            lines.append('\n')
                # other parameters
            for param in input_params.keys():
                if isinstance(input_params[param], (int, float)):
                    input_params[param] = str(input_params[param])
                else:
                    assert isinstance(input_params[param], type("")), f"The input paramter {param}={input_params[param]} must be string or number!"
                lines.append(f'{param} {input_params[param]}\n')
                #lines.append(f'\n')
            with open(filepath, 'w') as fl:
                fl.writelines(lines)

def from_abinit_to_ase(abopath):
	    
