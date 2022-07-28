
# opens a file with 1000 events and processes data to a neural network friendly format

import h5py
import sys
import torch
import os
import numpy as np
from tqdm import tqdm
# get home directory path
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()
# check 48 for error
# if __name__ == '__main__':
#     file = h5py.File("/work/submit/cfalor/upuppi/z_reg/train/notr/samples_v0_dijet_48.h5", "r")
#     file_out = h5py.File("/work/submit/cfalor/upuppi/z_reg/train/raw/samples_v0_dijet_48.h5", "w")
for fileid in range(1, 100):
#     if fileid == 40:
#         file = h5py.File("/work/submit/cfalor/upuppi/z_reg/train/notr/samples_v0_dijet_4.h5", "r")
#         # make a new file to store the processed data
#         file_out = h5py.File("/work/submit/cfalor/upuppi/z_reg/train/raw/samples_v0_dijet_4.h5", "w")
    # if fileid == 56:
    #     file = h5py.File("/work/submit/cfalor/upuppi/z_reg/test/notr/samples_v0_dijet_5.h5", "r")
    #     # make a new file to store the processed data
    #     file_out = h5py.File("/work/submit/cfalor/upuppi/deepjet-geometric/test/raw/samples_v0_dijet_5.h5", "w")
#     else:
    try:
        # file = h5py.File('/work/submit/bmaier/upuppi/data/v0_z_regression_pu30/test/raw/samples_v0_dijet_'+str(fileid)+".h5", "r")
        file = h5py.File(home_dir + 'all_data/raw/samples_v0_dijet_'+str(fileid)+".h5", "r")
        # file_out = h5py.File(home_dir + 'test2/raw/samples_v0_dijet_'+str(fileid)+".h5", "w")
        file_out = h5py.File(home_dir + 'all_data6/raw/samples_v0_dijet_'+str(fileid)+".h5", "w")
    except FileNotFoundError or OSError:
        # print the error
        print("fileid:", fileid)
        continue
    except OSError as e:
        print("fileid:", fileid)
        print(e)
        continue
    # copy the header from the original file
    

    # process the data
    # pfs: coordinates of particles in the event
    # pfs_shape: (1000, 7000, 7)
    # pt, eta, phi, E, pid, charge, z-position for pfs

    with file as f:
        pfs = f["pfs"][:, :1000, :]
        vtx = f["vtx"][:, :50]
        truth = f["truth"][:, :1000]
        z = f["z"][:, :1000]

        # for each event, get the max truth value
        max_truth = truth.max(axis=1)
        vtx_pt = vtx[:, 0, -1]
        vtx_sum_pt = vtx[:, :, -1].sum(axis=1)
        # discard events with max truth < 2
        nice_event_idx = ((max_truth >= 2) & (vtx_pt <= 40) & (vtx_sum_pt >= 0))
        pfs = pfs[nice_event_idx]
        vtx = vtx[nice_event_idx]
        truth = truth[nice_event_idx]
        z = z[nice_event_idx]


        # one hot encode the pid
        # pid takes values 0, 1, 2, 3, 4, -13/13
        # pid takes values +- 11, 13, 22, 130, 211, 321, 2112, 2212
        # pid_onehot takes values 0, 1, 2, 3, 4, 5, 6, 7 respectively (one hot encoded)
        # initialize the pid_onehot array
        pid = np.zeros((pfs.shape[0], pfs.shape[1], 6))
        for i in tqdm(range(pfs.shape[0])):
            for j in range(pfs.shape[1]):
                particle_id = int(pfs[i,j,4])
                if particle_id <= -13 or particle_id >= 13:
                    pid[i,j,5] = 1
                else:
                    try:
                        pid[i,j,particle_id] = 1
                    except:
                        print(i, j, particle_id)
                        raise(Exception("error"))
                # particle_id = abs(particle_id)
                # if particle_id == 11:
                #     pid[i,j,0] = 1
                # elif particle_id == 13:
                #     pid[i,j,1] = 1
                # elif particle_id == 22:
                #     pid[i,j,2] = 1
                # elif particle_id == 130:
                #     pid[i,j,3] = 1
                # elif particle_id == 211:
                #     pid[i,j,4] = 1
                # elif particle_id == 321:
                #     pid[i,j,5] = 1
                # elif particle_id == 2112:
                #     pid[i,j,6] = 1
                # elif particle_id == 2212:
                #     pid[i,j,7] = 1
        # convert pt, phi to cartesian coordinates
        # pt: (1000, 7000)
        # phi: (1000, 7000)
        # px: (1000, 7000)
        # py: (1000, 7000)
        # convert torch tensor to numpy array
        pt = pfs[:,:,0]
        eta = pfs[:,:,1]
        phi = pfs[:,:,2]
        E = pfs[:,:,3]
        q = pfs[:,:,5]
        z_data = pfs[:,:,6]
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        # replace all 0 E, pt with 1
        E[E==0] = 1
        pt[pt==0] = 1
        assert(np.all(E > 0))
        assert(np.all(pt > 0))
        log_E = np.log(E)
        log_pt = np.log(pt)


        # save the processed data in pfs
        # new_pfs = np.concatenate((px[:,:,np.newaxis], py[:,:,np.newaxis], eta[:,:,np.newaxis], log_E[:,:,np.newaxis], log_pt[:,:,np.newaxis], pid[:,:,:], q[:,:,np.newaxis], z[:,:,np.newaxis]), axis=2)
        # new features: px, py, eta, log_E, log_pt, q, pid, z
        # new features: px, py, eta, E, pid, q, z
        # new_pfs = np.concatenate((px[:,:,np.newaxis], py[:,:,np.newaxis], eta[:,:,np.newaxis], log_E[:,:,np.newaxis], pid[:,:,:], q[:,:,np.newaxis], z_data[:,:,np.newaxis]), axis=2)
        # new features: px, py, eta, phi, pt, E, log_E, pid, q, z
        new_pfs = np.concatenate((px[:,:,np.newaxis], py[:,:,np.newaxis], eta[:,:,np.newaxis], phi[:,:,np.newaxis], pt[:,:,np.newaxis], E[:,:,np.newaxis], log_E[:,:,np.newaxis], pid[:,:,:], q[:,:,np.newaxis], z_data[:,:,np.newaxis]), axis=2)
    # save the processed data in the file
    
    print("new_pfs shape:", new_pfs.shape)
    file_out.create_dataset("pfs", data=new_pfs)
    file_out.create_dataset("vtx", data=vtx)
    file_out.create_dataset("truth", data=truth)
    file_out.create_dataset("z", data=z)
    file_out.create_dataset('n', data=pfs.shape[0], dtype='i8')


    # save the file
    file_out.close()

    # close the original file
    file.close() 