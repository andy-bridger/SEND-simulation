from ase.io import read
from ase.io import write
from ase import Atoms
from ase.build import surface
from ase.build import fcc100
from ase.build import fcc110
from ase.build import fcc111
from ase.build import bulk
from matplotlib import pyplot as plt
from abtem import show_atoms
import abtem
import numpy as np
from skimage.measure import label
from math import isqrt
import hyperspy.api as hs

import h5py

with h5py.File('/dls/tmp/dto55534/au_al2cu_dual_precip.hdf5', 'r') as f:
    pix_data = f['array'][()]

d = hs.signals.Signal2D(pix_data)

d

d.plot(vmax='90th')

dose = 1e8
factor = dose / np.sum(d.data)

data_highD = factor * d.data
data_highD = np.random.poisson(data_highD)

data_highD = hs.signals.Signal2D(data_highD)

data_highD.save('/dls/tmp/dto55534/au_al2cu_dual_sim_dataset.hspy')
