
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

pop_top_vtx = True

for fileid in range(40, 50):
    try:
        file = h5py.File('/work/tier3/bmaier/upuppi/samples/events_'+str(fileid)+".h5", "r")
        # file_out = h5py.File(home_dir + 'test2/raw/samples_v0_dijet_'+str(fileid)+".h5", "w")
        file_out = h5py.File(home_dir + 'all_new_data/raw/samples_v0_dijet_'+str(fileid)+".h5", "w")
    except FileNotFoundError or OSError:
        # print the error
        print("fileid:", fileid)
        continue
    except OSError as e:
        print("fileid:", fileid)
        print(e)
        continue

# pfs = np.stack((final_pfs['pt'], final_pfs['eta'], final_pfs['phi'], final_pfs['e'], final_pfs['pid'], final_pfs['reco_z'], final_pfs['charge'], final_pfs['reco_z_vtx']), axis=-1)
# vtx = np.stack((final_vtx['x'], final_vtx['y'], final_vtx['z'], final_vtx['ndf'], final_vtx['total_pt']), axis=-1)

    with file as f:
        pfs = f["pfs"][:, :3000, :]
        vtx = f["vtx"][:, :50]
        truth = f["truth"][:, :3000]
        z = f["z"][:, :3000]
        # clamp the z values to be between -200 and 200
        z[z > 200] = 200
        z[z < -200] = -200
        

        if pop_top_vtx:
            num_pfs = pfs.shape[1]
            new_truth = np.zeros((truth.shape[0], truth.shape[1]))
            new_truth -= 99
            new_pfs = np.zeros((pfs.shape[0], pfs.shape[1], pfs.shape[2]))
            new_z = np.zeros((z.shape[0], z.shape[1]))
            vtx = vtx[:, 1:, :]
            for i in range(vtx.shape[0]):
                pileup_idx = (truth[i, :] != 0)
                high_pt_idx = (pfs[i, :, 0] > 1)
                keep_idx = np.logical_and(pileup_idx, high_pt_idx)
                pileup_idx = keep_idx
                if np.sum(truth[i] == 1) == 0:
                    print(truth[i])
                    print(vtx[i, :, :])
                    print(i, fileid)
                    raise(Exception("no truth value 1"))

                new_truth[i] = np.concatenate((truth[i][pileup_idx], np.zeros(num_pfs - np.sum(pileup_idx))), axis=0)
                new_truth[i] = new_truth[i] - 1
                new_pfs[i] = np.concatenate((pfs[i][pileup_idx], np.zeros((num_pfs - np.sum(pileup_idx), new_pfs.shape[2]))), axis=0)
                new_z[i] = np.concatenate((z[i][pileup_idx], np.zeros((num_pfs - np.sum(pileup_idx)))), axis=0)
            truth = new_truth
            pfs = new_pfs
            z = new_z

        # for each event, get the max truth value
        max_truth = truth.max(axis=1)
        nice_event_idx = (max_truth >= 2)
        pfs = pfs[nice_event_idx]
        vtx = vtx[nice_event_idx]
        truth = truth[nice_event_idx]
        z = z[nice_event_idx]


        # one hot encode the pid
        # pid takes values 0, 1, 2, 3, 4, -13/13
        # pid takes values +- 11, 13, 22, 130, 211, 321, 2112, 2212
        # pid_onehot takes values 0, 1, 2, 3, 4, 5, 6, 7 respectively (one hot encoded)
        # initialize the pid_onehot array
        pid = np.zeros((pfs.shape[0], pfs.shape[1], 8))
        for i in tqdm(range(pfs.shape[0])):
            for j in range(pfs.shape[1]):
                particle_id = int(pfs[i,j,4])
                try:
                    pid[i,j,particle_id] = 1
                except:
                    print(i, j, particle_id)
                    raise(Exception("error"))
        pt = pfs[:,:,0]
        eta = pfs[:,:,1]
        phi = pfs[:,:,2]
        E = pfs[:,:,3]
        z_pfc = pfs[:,:,5]
        q = pfs[:,:,6]
        z_vtx = pfs[:,:,7]
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
        # new features: px, py, eta, phi, pt, E, log_E, pid, z_pfc, q, z_vtx
        new_pfs = np.concatenate((px[:,:,np.newaxis], py[:,:,np.newaxis], eta[:,:,np.newaxis], phi[:,:,np.newaxis], pt[:,:,np.newaxis], E[:,:,np.newaxis], log_E[:,:,np.newaxis], z_pfc[:,:, np.newaxis], pid[:,:,:], q[:,:,np.newaxis], z_vtx[:,:,np.newaxis]), axis=2)
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