
# opens a file with 1000 events and processes data to a neural network friendly format

import h5py
import numpy as np
from tqdm import tqdm
import time
# get home directory path
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()

for fileid in range(1, 47):
    try:
        file = h5py.File('/work/submit/bmaier/upuppi/data/v0_z_regression_pu30/train/raw/samples_v0_dijet_'+str(fileid)+".h5", "r")
        file_out = h5py.File(home_dir + 'train2/raw/samples_v0_dijet_'+str(fileid)+".h5", "w")
    except FileNotFoundError or OSError:
        # print the error
        print("fileid:", fileid)
        continue
    except OSError as e:
        print("fileid:", fileid)
        print(e)
        continue
    # copy the header from the original file
    # add an additional feature to vtx
    for key in file.keys():
        if key == "z" or key == "pfs":
            file_out.create_dataset(key, data=file[key][:])
        elif key=="n":
            # it is the number of events, scalar
            file_out.create_dataset(key, data=file[key])

    with file as f:
        # print(f.keys())
        vtx = f["vtx"][:]
        pfs = f["pfs"][:]
        truth = f["truth"][:]
        n = f["n"]
        new_vtx = np.zeros((vtx.shape[0], vtx.shape[1], vtx.shape[2]+1))
        new_truth = np.zeros((truth.shape[0], truth.shape[1]))
        new_truth -= 99

        # add an additional feature to vtx, total pt
        # for each event in the file, add the total pt of the particles in the event
        for i in tqdm(range(vtx.shape[0])):
            for j in range(vtx.shape[1]):
                # get the indices which have truth value j
                indices = np.where(truth[i] == j)
                # add the pt of the particles in the event
                # check if indices is empty
                if len(indices[0]) == 0:
                    new_vtx[i, j, -1] = 0
                else:
                    pfc_pt = pfs[i, indices, 0]
                    vtx_pt = np.sum(pfc_pt)
                    if vtx_pt < 0:
                        # get the particles with pt < 0
                        indices_neg = np.where(pfs[i][indices][0] < 0)
                        # get the pt of the particles with pt < 0
                        pfc_neg = pfs[i][indices][0][indices_neg]
                        raise(Exception("vtx_pt < 0"))
                    new_vtx[i][j][-1] = vtx_pt
                # copy the coordinates of the vertices
                new_vtx[i][j][:-1] = vtx[i][j][:]
            # stable sort new_vtx by highest total pt and update the truth 
            sorted_indices = np.lexsort((-new_vtx[i][:, -2], -new_vtx[i][:, -1]))
            truth_indices = np.argsort(sorted_indices)
            new_vtx[i] = new_vtx[i][sorted_indices]
            for c, t in enumerate(truth[i]):
                # convert t to int
                t = int(t)
                if t <= 0:
                    new_truth[i][c] = t
                else:
                    new_truth[i][c] = truth_indices[t]
            # print("new_vtx_sorted:", new_vtx_sorted)
            # print("new_truth:", new_truth[i])
        file_out.create_dataset("vtx", data=new_vtx)
        file_out.create_dataset("truth", data=new_truth) 
    print(file_out.keys())
    file_out.close()
    file.close()
    print("fileid:", fileid)
    print("done")