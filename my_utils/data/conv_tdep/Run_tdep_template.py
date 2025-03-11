import numpy as np
from pathlib import Path

from ase.io import read, write
from ase.build import make_supercell

from pytdep.runtdep import RunTdep


wdir
folderbin
nproc
refine_cell
nthrow
rc2
rc3
U0
qg
traj_path
uc_path
mult_mat
temperature
index



# LINE 28; DON'T WRITE ANYTHING ABOVE IT

wdir = Path(wdir)

uc = read(uc_path)

sc = make_supercell(uc, mult_mat)

traj = read(traj_path, index=':')[:index]

# TDEP
runtdp = RunTdep(uc,
                 sc,
                 traj,
                 temperature=temperature, 
                 folder=wdir.absolute(), 
                 folderbin=folderbin,
                 nproc=nproc,
                 refine_cell=refine_cell,
                 nthrow=nthrow)

runtdp.run(options_forceconstant=dict(rc2 = rc2,
                                      rc3 = rc3,
                                      U0 = U0),
           options_unitcell = dict(rc2 = rc2),
           options_phonons=dict(qg = qg))


