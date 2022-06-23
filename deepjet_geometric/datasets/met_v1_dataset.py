import os.path as osp
import glob

import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class MetV1(Dataset):
    r'''
        input x0: (PFCandidates)
               
        input x1: (ClusterInfo)

        input x2: (GlobalInfo) 
        
        # truth target = hard energy fraction
    '''

    url = '/dummy/'

    def __init__(self, root, transform=None):
        super(MetV1, self).__init__(root, transform)
        
        self.strides = [0]
        self.calculate_offsets()

    def calculate_offsets(self):
        for path in self.raw_paths:
            with h5py.File(path, 'r') as f:
                self.strides.append(f['n'][()][0])
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
        edge_index = torch.empty((2,0), dtype=torch.long)
        with h5py.File(self.raw_paths[file_idx]) as f:
            
            Npfc = np.count_nonzero(f['x0'][idx_in_file,:,0])
            Nclus = np.amax(f['x1'][idx_in_file,:,0]) #getting the maximum cluster_idx

            x_pfc = f['x0'][idx_in_file,:Npfc,:]
            x_clus = f['x1'][idx_in_file,:int(Nclus),:]
            x_glob = f['x2'][idx_in_file,:Npfc,:]

            # convert to torch
            x_pfc = torch.from_numpy(x_pfc)
            x_clus = torch.from_numpy(x_clus)
            x_glob = torch.from_numpy(x_glob)

            # target
            y = torch.from_numpy(f['y0'][idx_in_file,:Npfc,0])

            # gen info
            #gen = f['geninfo']
            
            #return Data(x=x_pfc, edge_index=edge_index, y=y,
            #            x_pfc=x_pfc, x_clus=x_clus, x_glob=x_glob, gen=gen, N=Npfc)
            
            return Data(x=x_pfc, edge_index=edge_index, y=y,
                        x_pfc=x_pfc, x_clus=x_clus, x_glob=x_glob)

