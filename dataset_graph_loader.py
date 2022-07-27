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
            # connect edges between all same truth
            edges = np.zeros((2, 0), dtype=np.int64)
            # for i in range(Npfc):
            #     for j in range(i+1, Npfc):
            #         edges = np.concatenate((edges, np.array([[i], [j]])), axis=1)
            # choose edges randomly
            # if Npfc*(Npfc-1)//2 < 11000:
            if True:
                for i in range(Npfc):
                    for j in range(i+1, Npfc):
                        edges = np.concatenate((edges, np.array([[i], [j]])), axis=1)
            else:
                edges1 = np.random.choice(Npfc, size=(1, int(10000)))
                edges2 = np.random.choice(Npfc, size=(1, int(10000)))
                edges = np.concatenate((edges1, edges2), axis=0)
                edges = np.unique(edges, axis=1)
                edges = edges[:,edges[0] != edges[1]]
            edges = torch.from_numpy(edges).long()
            # make it undirected
            edges = to_undirected(edges)
            # remove self-loops
            # remove duplicate edges

            # gen info
            #gen = f['geninfo']
            
            #return Data(x=x_pfc, edge_index=edge_index, y=y,
            #            x_pfc=x_pfc, x_clus=x_clus, x_glob=x_glob, gen=gen, N=Npfc)
            
            return Data(x=x_pfc, edge_index=edges, y=y,
                        x_pfc=x_pfc, x_vtx=x_vtx, truth=truth)
