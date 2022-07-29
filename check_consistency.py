
# opens a file with 1000 events and processes data to a neural network friendly format

import h5py
import numpy as np
from tqdm import tqdm
import time
# get home directory path
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()



for fileid in range(1, 101):
    try:
        file = h5py.File('/work/tier3/bmaier/upuppi/samples/events_'+str(fileid)+".h5", "r")
        # file = h5py.File('/work/submit/cfalor/upuppi/datapreparation/h5_gen/test.h5', "r")
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

    with file as f:
        # print(f.keys())
        vtx = f["vtx"][:]
        pfs = f["pfs"][:]
        truth = f["truth"][:]
        n = f["n"][()]
        for i in range(n):
            vtx_event = vtx[i]
            pfs_event = pfs[i]
            truth_event = truth[i]
            Nvtx = np.count_nonzero(vtx_event[:, 0])
            Npfs = np.count_nonzero(pfs_event[:, 0])
            for j in range(Nvtx):
                if np.sum(truth_event == j) != vtx_event[j,-2]:
                    print(j, np.sum(truth_event == j), vtx_event[j,-2])
                    print(truth_event[:Npfs])
                    print(vtx_event[:Nvtx])
                    raise(Exception("error"))
                elif np.sum(truth[i] == 1) == 0:
                    print(i, truth[i])
                    print(vtx_event[:Nvtx])
                    raise(Exception("error"))
