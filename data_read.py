import h5py
import sys
# import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

# import uproot as uproot
# import smear_chargedhadrons as sch
# from awkward import Array
# from awkward.layout import ListOffsetArray64

file = h5py.File("/work/submit/cfalor/upuppi/deepjet-geometric/train/raw/samples_v0_dijet_46.h5", "r")

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
# vtx_shape: (1000, 200, 4)tr
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
# z coordinate of each particle
z = events["z"][:]

truth = events["truth"][:]
# max of truth
max_truth = np.max(truth)
print("max_truth:", max_truth)
# print out the shape of the data
print("pfs shape:", pfs.shape)
print(pfs[960,1047,:])
print(pfs[9,107,:])

# after processing the data, the shape of the data is (1000, 7000, 13)
# with columns px, py, eta, E, one-hot encoded pid, charge, z-position
# print out a sample of the data
# print("pfs sample:", pfs[0,0:10,:])

pid = pfs[:,:,6]


valid_truth_idx = truth > 0
pid = pid[valid_truth_idx]
z = z[valid_truth_idx]

electron_idx = np.where(pid == 1)
print("electron_idx:", electron_idx)
z_electron = z[electron_idx]
# plot the z-coordinate of electrons
# mean and standard deviation of the z-coordinate of electrons
mean_electron = np.mean(z_electron)
std_electron = np.std(z_electron)
print("mean_electron:", mean_electron)
print("std_electron:", std_electron)
plt.hist(z_electron, bins=100)
# add mean and standard deviation of the z-coordinate of electrons to the plot
plt.axvline(mean_electron, color='r', linestyle='dashed', linewidth=2)
plt.axvline(mean_electron + std_electron, color='r', linestyle='dashed', linewidth=2)
plt.axvline(mean_electron - std_electron, color='r', linestyle='dashed', linewidth=2)
plt.title("z-coordinate of electrons")
plt.xlabel("z-coordinate")
plt.ylabel("count")
# save the figure
plt.savefig("z_electron.png")
phi = pfs[:,:,2]
z_input = pfs[:,:,3]
# print(z_input[4,0:10])
# print(z[4, 0:10])
# print(np.min(phi))
# print(pid.shape)
# pid = pid.astype(int)
# # # print all unique values in the pid column
# # print("unique pid values:", np.unique(pid))
# vnum = vtx[:,:,-1]
# vnum = vnum.astype(int)
# # print("unique vnum values:", np.unique(vnum))
# # # pid has the following values: unique pid values: [-13   0   1   2   3   4  13]
# # # print the number of particles with each pid
# # pid_flat = pid.flatten()
# # # add 13 to the pid to get the correct pid
# # pid_flat = pid_flat + 13
# # print("number of particles with each pid:", np.bincount(pid_flat))
# # maxpid = pid[27,343]
# # print("Max pid:", maxpid)
# # print("vtx shape:", vtx.shape)
# # print("z shape:", z.shape)
# # print(pfs[27,343,:])
# # print(z.shape)
# z = z[valid_truth_idx]
# z_input = z_input[valid_truth_idx]
# # print(z.shape)
# # flatten the z
# z = z.flatten()
# # remove 0 values from z
# z = z.astype(float)
# # # plot the z-coordinate of the particles
# z_input = z_input.flatten()
# # print(z_input[4000:4100])
# plt.hist(z, bins=100)
# plt.savefig("z_hist.png")
# plt.close()
# # flatten truth
# truth_flat = truth.flatten()
# # add 1 to the truth to get the correct truth
# truth_flat = truth_flat
# truth_flat = truth_flat.astype(int)
# # print unique values in the truth column
# # print("unique truth values:", np.unique(truth_flat))
# # print("number of particles with each truth:", np.bincount(truth_flat))

       