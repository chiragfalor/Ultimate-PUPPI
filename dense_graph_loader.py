import os.path as osp
import glob
from time import sleep

import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.utils import to_undirected

class UPuppiV0(Dataset):
    r'''
        input pfs: (PFCandidates)
               
        input vtx: (VertexInfo)
        
        # truth target = pf candidate z position
    '''

    url = '/dummy/'

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'vtx':
            return None
        elif key == 'pfs':
            return None
        elif key == 'x_pfc':
            return None
        else:
            return super(UPuppiV0, self).__cat_dim__(key, value, *args, **kwargs)

    def __init__(self, root, transform=None):
        super(UPuppiV0, self).__init__(root, transform)
        
        self.strides = [0]
        self.calculate_offsets()

    def calculate_offsets(self):
        for path in self.raw_paths:
            with h5py.File(path, 'r') as f:
                self.strides.append(f['n'][()])
        self.strides = np.cumsum(self.strides)

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.z files to {}'.format(self.url, self.raw_dir))

    def len(self):
        return self.strides[-1]

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.h5')))
        return raw_files

    @property
    def processed_file_names(self):
        return []


    def get(self, idx):
        file_idx = np.searchsorted(self.strides, idx) - 1
        idx_in_file = idx - self.strides[max(0, file_idx)] - 1
        if file_idx >= self.strides.size:
            raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        with h5py.File(self.raw_paths[file_idx], 'r') as f:
            
            Npfc = np.count_nonzero(f['pfs'][idx_in_file,:,0])
            Nvtx = np.count_nonzero(f['vtx'][idx_in_file,:,-1]) #getting the maximum cluster_idx

            x_pfc = f['pfs'][idx_in_file,:Npfc,:]
            x_vtx = f['vtx'][idx_in_file,:Nvtx,:]

            pfSelection = np.ones(Npfc, dtype=bool)
            #pfSelection = (x_pfc[:, -2] != 0)

            x_pfc = x_pfc[pfSelection, :]

            # convert to torch
            x_pfc = torch.from_numpy(x_pfc).float()
            x_vtx = torch.from_numpy(x_vtx).float()

            # target
            y = torch.from_numpy(f['z'][idx_in_file,:Npfc][pfSelection]).float()
            truth = torch.from_numpy(f['truth'][idx_in_file,:Npfc][pfSelection]).int()
            
            # get a completely connected graph
            adj = np.zeros((Npfc, Npfc), dtype=bool)
            # fill it with True for all entries
            adj[np.triu_indices(Npfc, 1)] = True
            adj[np.tril_indices(Npfc, -1)] = True
            # to torch
            adj = torch.from_numpy(adj).byte()
            


            return Data(x=x_pfc, adj = adj, y=y,
                        x_pfc=x_pfc, x_vtx=x_vtx, truth=truth)