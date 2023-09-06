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

## Make segmented space as microstructure

specimen = np.zeros((200, 200, 100), dtype=int)
specimen_coords = np.asarray(np.where(specimen == 0)).T

specimen_thickness = 0.75 #fractional_thickness

specimen[:,:,:int(specimen.shape[2] * specimen_thickness)] = 1

def view_specimen(specimen, sampling = (8,8,2), alpha = 0.15, s = 1):
    specimen_coords = np.asarray(np.where(specimen != -1)).T
    sc_shape = list(specimen.shape) + [3]

    specimen_coords = specimen_coords.reshape(sc_shape)
    specimen_coords = specimen_coords[::sampling[0], ::sampling[1], ::sampling[2]]
    specimen_coords = specimen_coords.reshape((specimen_coords[:,:,:,0].size, 3))
    
    flat_spec = specimen[::sampling[0], ::sampling[1], ::sampling[2]].reshape(-1)
    
    for uind in np.unique(specimen):
        phase_coords = specimen_coords[np.where(flat_spec == uind)]
        not_phase_coords = specimen_coords[np.where(flat_spec != uind)]
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(phase_coords[:,0], phase_coords[:,1], phase_coords[:,2], 
                   s = s)
        ax.scatter(not_phase_coords[:,0], not_phase_coords[:,1], not_phase_coords[:,2], 
                   s = s, alpha = alpha)
        ax.set_title(f'Phase: {uind}')
    

import copy

def alter_below_plane(specimen, normal, point, ind_to, inds_from):
    '''
    Will change all the voxel indices that match those specified in `inds_from` and are
    below the plane defined by `normal` and `point` to voxel index `ind_to`
    
    e.g. we start with a specimen which is a 3d array (X,Y,Z) of zeros.
    If we specify the plane with normal (0,0,1) {a plane parallel to the x,y plane} that 
    goes through point (50,50,50) and ind_from as 0 {all values of 0 will change} and ind_to 
    as 1. We will get returned a 3d array where all the values of [X,Y,:50] are 1 and everything else
    remains 0
    
    specimen: (3D array)
    normal: (3 value structure) a,b,c values of the surface normal
    point: (3 value structure) x,y,z values of a point on the plane
    ind_to: (ind) The index you want these specified voxels to become
    ind_from: (ind or iterable of inds) The index/indicies you would like to change, if they statisfy 
                the location condition
    '''
    
    new_specimen = copy.deepcopy(specimen)
    
    if type(inds_from) == int:
        inds_from = [inds_from]
        
    if type(inds_from) == type(None):
        inds_from = np.unique(new_specimen)
        print(inds_from)

    a,b,c = normal #surface normal
    x0,y0,z0 = point #point on plane

    mask = np.zeros_like(specimen, dtype=bool)
    for _x in np.arange(specimen.shape[0]):
        for _y in np.arange(specimen.shape[1]):
            for _z in np.arange(specimen.shape[2]):
                if (a * (_x - x0)  + b * (_y - y0) + c * (_z - z0)) <= 0 :
                    mask[_x, _y, _z] = 1
                    
    from_mask = np.sum([np.where(specimen ==i_from, 1, 0) for i_from in inds_from], axis = 0)

    mask = mask*from_mask

    new_specimen[np.where(mask==1)] = ind_to
    
    return new_specimen

def alter_within_sphere(specimen, centre, radius, ind_to, inds_from):
    
    '''
    Will change all the voxel indices that match those specified in `inds_from` and are
    within the sphere defined by `centre` and `radius` to voxel index `ind_to`
    
    
    specimen: (3D array)
    centre: (3 value structure) x,y,z values of a point at the centre of the sphere
    rdaius: (float) radius of the sphere
    ind_to: (ind) The index you want these specified voxels to become
    ind_from: (ind or iterable of inds) The index/indicies you would like to change, if they statisfy 
                the location condition
    '''
    
    new_specimen = copy.deepcopy(specimen)
    
    if type(inds_from) == int:
        inds_from = [inds_from]
        
    if type(inds_from) == type(None):
        inds_from = np.unique(new_specimen)
        print(inds_from)
        
    x0,y0,z0 = centre #point on plane

    mask = np.zeros_like(specimen, dtype=bool)
    for _x in np.arange(specimen.shape[0]):
        for _y in np.arange(specimen.shape[1]):
            for _z in np.arange(specimen.shape[2]):
                if ((_x - x0)**2  + (_y - y0)**2 + (_z - z0)**2) <= radius**2 :
                    mask[_x, _y, _z] = 1
                    
    from_mask = np.sum([np.where(specimen ==i_from, 1, 0) for i_from in inds_from], axis = 0)

    mask = mask*from_mask

    new_specimen[np.where(mask==1)] = ind_to
    
    return new_specimen

norm1 = (2,2,3) #surface normal
p1 = (100,100,50) #point on plane

new_specimen = alter_below_plane(specimen, norm1, p1, 2, 1)

view_specimen(new_specimen, sampling = (8,8,2))

norm2 = (2,1,2) #surface normal
p2 = (90,90,50) #point on plane

new_specimen2 = alter_below_plane(new_specimen, norm2, p2, 3, (1,2))

view_specimen(new_specimen2, sampling = (8,8,2))

norm3 = (0,0,1) #surface normal
p3 = (70,70,20) #point on plane

new_specimen3 = alter_below_plane(new_specimen2, norm3, p3, 3, 2)
new_specimen3 = alter_below_plane(new_specimen3, (0,0,1), (0,0,200), 1, 2) # make everything in the cell 2 -> 1

view_specimen(new_specimen3, sampling = (8,8,2))

#build a ledge into a second flat height

new_specimen4 = alter_below_plane(new_specimen3, (1,1,-1), (50,50,50), 0, 3)
new_specimen4 = alter_below_plane(new_specimen4, (0,0,1), (50,50,30), 3, 0)

view_specimen(new_specimen4, sampling = (8,8,2))

## add some surface precipitates

desired_expected_precip = 3
max_radius = 20
random_seed = 5
np.random.seed(random_seed)



air_fraction = np.where(new_specimen4 == 0,1,0).sum()/new_specimen4.size
air_cols = new_specimen4.shape[2]-np.argmin(new_specimen4, axis = -1)
prop_within_max_r_of_surface = np.mean((np.ones_like(air_cols)*max_radius)/air_cols)
prop_on_surface = 0.25*prop_within_max_r_of_surface

prop_of_eligable_precip = air_fraction*prop_on_surface

n_precip = int(np.ceil(desired_expected_precip/prop_of_eligable_precip)) #will end up getting 
print(n_precip)


##get some random centres within the scene
centres = np.random.random_sample((n_precip, 3), ) * np.array(specimen.shape)

##and some random radii
radii = np.random.random((n_precip))*max_radius

centres, radii


centre_approx_locations = np.round(centres, 0).astype(int)

good_cent = []
good_rad = []

for c_ind, c in enumerate(centre_approx_locations):
    if new_specimen4[c[0], c[1], c[2]] == 0:
        d = c[2] - np.argmin(new_specimen4[c[0], c[1]])
        c_rad = radii[c_ind]
        if d <= c_rad:          
            print(d, c[2], c_rad)
            good_cent.append(c)
            good_rad.append(c_rad)
        



specimen5 = copy.deepcopy(new_specimen4)
for sphere_i in range(len(good_cent)):
    if sphere_i%2 == 0:
        specimen5 = alter_within_sphere(specimen5, good_cent[sphere_i], good_rad[sphere_i], 2, [0,1,3])
    else:
        specimen5 = alter_within_sphere(specimen5, good_cent[sphere_i], good_rad[sphere_i], 4, [0,1,3])

view_specimen(specimen5, sampling=(8,8,2), s = 3, alpha=0.05)



labelled_specimen, num_labels = specimen5, len(np.unique(specimen5))

num_labels

np.unique(labelled_specimen)

labelled_specimen.shape

fig, ax = plt.subplots(3,2)
ax[0,0].imshow(labelled_specimen [:,:,0], cmap='nipy_spectral')
ax[0,1].imshow(labelled_specimen [:,:,80], cmap='nipy_spectral')
ax[1,0].imshow(labelled_specimen [:,0,:], cmap='nipy_spectral')
ax[1,1].imshow(labelled_specimen [:,50,:], cmap='nipy_spectral')
# ax[2,0].imshow(labelled_specimen [0,:,:], cmap='nipy_spectral')
# ax[2,1].imshow(labelled_specimen [-1,:,:], cmap='nipy_spectral')
ax[2,0].imshow(labelled_specimen [:,:,15], cmap='nipy_spectral')
ax[2,1].imshow(labelled_specimen [:,:,45], cmap='nipy_spectral')

phase1_num = np.where(labelled_specimen== 1,1,0).sum(axis = 2)
phase2_num = np.where(labelled_specimen== 2,1,0).sum(axis = 2)
phase3_num = np.where(labelled_specimen== 3,1,0).sum(axis = 2)
phase4_num = np.where(labelled_specimen== 4,1,0).sum(axis = 2)

from pathlib import Path

phase_path = '/dls/tmp/dto55534/SimulatedData'

np.save(phase_path+'/fcc110Au_sum.npy', phase1_num)
np.save(phase_path+'/Al2Cu_or1_sum.npy', phase2_num)
np.save(phase_path+'/fcc100Au_sum.npy', phase3_num)
np.save(phase_path+'/Al2Cu_or2_sum.npy', phase4_num)

plt.figure()
plt.imshow(phase1_num)
plt.savefig(phase_path+'/fcc110Au_sum.jpg')

plt.figure()
plt.imshow(phase2_num)
plt.savefig(phase_path+'/Al2Cu_or1_sum.jpg')

plt.figure()
plt.imshow(phase4_num)
plt.savefig(phase_path+'/Al2Cu_or2_sum.jpg')

plt.figure()
plt.imshow(phase3_num)
plt.savefig(phase_path+'/fcc100Au_sum.jpg')

phase_proportion = phase1_num/(phase1_num+phase3_num)

plt.figure()
plt.imshow(phase_proportion)



abtem.__version__

from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

with MPRester("api-key") as mpr:
    docs = mpr.summary.search(material_ids=["mp-134"], fields=["structure"])
    al = AseAtomsAdaptor().get_atoms(docs[0].structure)
    
    docs2 = mpr.summary.search(material_ids=["mp-985806"], fields=["structure"])
    al2cu = AseAtomsAdaptor().get_atoms(docs2[0].structure)

# view(al, viewer='nglview')
# show_atoms(al);


# show_atoms(al2cu);


## This bit is for if you need to create a large rotated bulk of atoms, just so you can quickly check that you've created a bulk that covers the entire sample area

cube_corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], ])

cube_corners

cube_corners * np.array((200,200,100))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull



def show_extent(pts, cube_size):
    cube_corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], ]) * np.array(cube_size)
    
    
    cube_hull = ConvexHull(cube_corners)

    hull = ConvexHull(pts)
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection="3d")

    # Plot defining corner points
    #ax.plot(pts.T[0], pts.T[1], pts.T[2], "ko")

    # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")
    
    for s in cube_hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(cube_corners[s, 0], cube_corners[s, 1], cube_corners[s, 2], "b-")

    # Make axis label
    for i in ["x", "y", "z"]:
        eval("ax.set_{:s}label('{:s}')".format(i, i))

    plt.show()

ppt1 = al2cu.repeat((100,80,100))
ppt1.translate([0,0,100])
ppt1.rotate(30, 'x')

ppt1.positions

show_extent(ppt1.positions, (200,200,100))




# ppt1.translate([0,0,20])

 # Get all the postive positions!
# ppt2 = ppt1[[all(ppt1.positions[i,:]) for i in range(ppt1.positions.shape[0])]]
ppt2 = ppt1[[ppt1.positions[i,0] > 0 for i in range(ppt1.positions.shape[0])]]
ppt2_y = ppt2[[ppt2.positions[i,1] > 0 for i in range(ppt2.positions.shape[0])]]
ppt2_z = ppt2_y[[ppt2_y.positions[i,2] > 0 for i in range(ppt2_y.positions.shape[0])]]

ppt2_z.positions.shape


# gr2 = fcc100('Al', size=(100,100,20))
# gr2.translate([-100., 0 , 0])
ppt2_pos_int = ppt2_z.positions.astype(int)
label = 2
mask_atoms = np.zeros(ppt2_pos_int.shape[0], dtype=bool)
for _ in np.arange(ppt2_pos_int.shape[0]):
    if ppt2_pos_int[_,0] < labelled_specimen.shape[0]:
        if ppt2_pos_int[_,1] < labelled_specimen.shape[1]:
            if ppt2_pos_int[_,2] < labelled_specimen.shape[2]:
                if labelled_specimen[ppt2_pos_int[_,0], ppt2_pos_int[_,1], ppt2_pos_int[_,2]] == label:
                    mask_atoms[_] = 1
ppt_label_3 = ppt2_z[mask_atoms]

fig, axs = plt.subplots(1,3)

abtem.show_atoms(ppt_label_3, plane='xz', ax=axs[0])
abtem.show_atoms(ppt_label_3, plane='xy', ax=axs[1])
abtem.show_atoms(ppt_label_3, plane='yz', ax=axs[2])



ppt1 = al2cu.repeat((100,80,60))
ppt1.translate([0,0,100])
#ppt1.rotate(30, 'x')

ppt1.positions

show_extent(ppt1.positions, (200,200,100))

# ppt1.translate([0,0,20])

 # Get all the postive positions!
# ppt2 = ppt1[[all(ppt1.positions[i,:]) for i in range(ppt1.positions.shape[0])]]
ppt2 = ppt1[[ppt1.positions[i,0] > 0 for i in range(ppt1.positions.shape[0])]]
ppt2_y = ppt2[[ppt2.positions[i,1] > 0 for i in range(ppt2.positions.shape[0])]]
ppt2_z = ppt2_y[[ppt2_y.positions[i,2] > 0 for i in range(ppt2_y.positions.shape[0])]]

ppt2_z.positions.shape


# gr2 = fcc100('Al', size=(100,100,20))
# gr2.translate([-100., 0 , 0])
ppt2_pos_int = ppt2_z.positions.astype(int)
label = 4
mask_atoms = np.zeros(ppt2_pos_int.shape[0], dtype=bool)
for _ in np.arange(ppt2_pos_int.shape[0]):
    if ppt2_pos_int[_,0] < labelled_specimen.shape[0]:
        if ppt2_pos_int[_,1] < labelled_specimen.shape[1]:
            if ppt2_pos_int[_,2] < labelled_specimen.shape[2]:
                if labelled_specimen[ppt2_pos_int[_,0], ppt2_pos_int[_,1], ppt2_pos_int[_,2]] == label:
                    mask_atoms[_] = 1
ppt_label1 = ppt2_z[mask_atoms]

fig, axs = plt.subplots(1,3)

abtem.show_atoms(ppt_label1, plane='xz', ax=axs[0])
abtem.show_atoms(ppt_label1, plane='xy', ax=axs[1])
abtem.show_atoms(ppt_label1, plane='yz', ax=axs[2])



gr1 = fcc110('Au', size=(200,200,100))
gr1_pos_int = gr1.positions.astype(int)
label = 1
mask_atoms = np.zeros(gr1_pos_int.shape[0], dtype=bool)
for _ in np.arange(gr1_pos_int.shape[0]):
    if gr1_pos_int[_,0] < labelled_specimen.shape[0]:
        if gr1_pos_int[_,1] < labelled_specimen.shape[1]:
            if gr1_pos_int[_,2] < labelled_specimen.shape[2]:
                if labelled_specimen[gr1_pos_int[_,0], gr1_pos_int[_,1], gr1_pos_int[_,2]] == label:
                    mask_atoms[_] = 1
gr1_mod = gr1[mask_atoms]


gr2 = fcc100('Au', size=(200,200,100))
# gr2.translate([-100., 0 , 0])
gr2_pos_int = gr2.positions.astype(int)
label = 3
mask_atoms = np.zeros(gr2_pos_int.shape[0], dtype=bool)
for _ in np.arange(gr2_pos_int.shape[0]):
    if gr2_pos_int[_,0] < labelled_specimen.shape[0]:
        if gr2_pos_int[_,1] < labelled_specimen.shape[1]:
            if gr2_pos_int[_,2] < labelled_specimen.shape[2]:
                if labelled_specimen[gr2_pos_int[_,0], gr2_pos_int[_,1], gr2_pos_int[_,2]] == label:
                    mask_atoms[_] = 1
gr2_mod = gr2[mask_atoms]


sample_model = gr2_mod + gr1_mod + ppt_label1 + ppt_label_3

fig, axs = plt.subplots(3,1, figsize = (5, 12))
abtem.show_atoms(sample_model, plane = 'xz', ax=axs[0])
abtem.show_atoms(sample_model, plane = 'yz', ax=axs[1])
abtem.show_atoms(sample_model, plane = 'xy', ax=axs[2])

fig = plt.gcf()
size = fig.get_size_inches()


import ase


## Make 4DSTEM data

np.max(np.asanyarray(sample_model.positions))

sample_model.cell = [200, 200, 100]

potential = abtem.Potential(
    sample_model,
    sampling=0.05,
    parametrization="lobato",
    slice_thickness=1,
    projection="finite",
    device = 'gpu'
)

probe = abtem.Probe(
    energy=300e3,
    semiangle_cutoff=1.5)
probe.grid.match(potential)

intensity = probe.build()



intensity.show()

scan_start = [20.0, 20.0]
scan_end = [180.0, 180.0]

scan_start_arr = np.asarray(scan_start)
scan_end_arr = np.asarray(scan_end)

#assuming cubic cell
cell_a = sample_model.cell[0][0]
cell_b = sample_model.cell[1][1]

scan_start_frac = scan_start_arr/np.array((cell_a, cell_b))
scan_end_frac = scan_end_arr/np.array((cell_a, cell_b))

# grid_scan = abtem.GridScan.from_fractional_coordinates(
#     potential,
#     start=[0, 0],
#     end=[1 / 3, 1 / 2],
#     sampling=probe.aperture.nyquist_sampling,
# )

gridscan = abtem.GridScan(start=scan_start, end=scan_end, sampling=[2.0, 2.0])

ax, im = potential.project().show();

gridscan.add_to_mpl_plot(ax)

probe.antialias_aperture

probe.cutoff_scattering_angles

probe.show(cmap='inferno', power=.3);

from abtem.detect import PixelatedDetector

pixelated_detector = PixelatedDetector(max_angle=50.0)

pixelated_measurement = probe.scan(gridscan, pixelated_detector, potential)

pixelated_measurement.write('/dls/tmp/dto55534/au_al2cu_dual_precip.hdf5')
