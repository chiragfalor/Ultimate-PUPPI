import h5py
import sys
import time
# import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader

from tqdm import tqdm
# import uproot as uproot
# import smear_chargedhadrons as sch
# from awkward import Array
# from awkward.layout import ListOffsetArray64
# get home directory path
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()

file = h5py.File(home_dir + 'train/raw/samples_v0_dijet_28.h5', "r")

print("Keys:", file.keys())

events = file

# edge_start: start particle of the edge
# edge_start_shape: (1000, 7000)
# edge_stop: stop vertex of the edge (same as vtx_truthidx)
# edge_stop_shape: (1000, 200)
# n: number of snapshots/events = 1000
# pfs: coordinates of particles in the event
# pfs_shape: (1000, 7000, 7)
# pt, eta, phi, E, pid, charge, z-position for pfs
# truth: The true vertex where the particle was produced. Contains the vertex id from 1-200
# truth_shape: (1000, 7000)
# vtx: The coordinates of the vertices of the event
# vtx_shape: (1000, 200, 4)
# x, y, z, #particles for vtx
# vtx_truthidx: lists out indices which are actually vertices, otherwise 0
# vtx_truthidx_shape: (1000, 200)
# z: z-coordinate of each particle
# z_shape: (1000, 7000)



#Keys: <KeysViewHDF5 ['edge_start', 'edge_start_shape', 'edge_stop', 'edge_stop_shape', 'n', 'pfs', 'pfs_shape', 'truth', 'truth_shape', 'vtx', 'vtx_shape', 'vtx_truthidx', 'vtx_truthidx_shape', 'z', 'z_shape']>
# check the type of columns in the file

#pfs
pfs = events["pfs"][:]
#vtx
vtx = events["vtx"][:]
print(vtx)
# z coordinate of each particle
z = events["z"][:]

truth = events["truth"][:]
# max of truth
print(truth.shape)
max_truth = np.max(truth, axis=1)
print(max_truth.shape)
print("max_truth:", max_truth)
# print out the shape of the data
print("pfs shape:", pfs.shape)
print(pfs[960,1047,:])
print(pfs[9,107,:])

# after processing the data, the shape of the data is (1000, 7000, 13)
# with columns px, py, eta, E, one-hot encoded pid, charge, z-position
# print out a sample of the data
# print("pfs sample:", pfs[0,0:10,:])

pid = pfs[:,:,4]


valid_truth_idx = truth > 0
pid = pid[valid_truth_idx]
z = z[valid_truth_idx]

# print(z_input[4,0:10])
# print(z[4, 0:10])
# print(np.min(phi))
print(pid.shape)
pid = pid.astype(int)
# # # print all unique values in the pid column
print("unique pid values:", np.unique(pid))
# print the histogram of number of particles for each pid
print("histogram of pid:", np.histogram(pid, bins=np.unique(pid)))

BATCHSIZE = 1
start_time = time.time()
data_train = UPuppiV0(home_dir + 'train2/')
data_test = UPuppiV0(home_dir + 'test2/')

print(data_train)
print(data_test)


print(type(data_train))


data_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])

# print 5 random samples from the dataset

for batch_idx, data in enumerate(tqdm(data_loader)):
    break
    print(data)
    print(data.x_pfc.shape)
    print(data.x_vtx)
    print(data.x_pfc_batch)
    print(data.y)
    print(batch_idx)
    if batch_idx == 5:
        break