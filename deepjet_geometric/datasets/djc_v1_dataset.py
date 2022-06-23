import os.path as osp
import glob

import h5py
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class DeepJetCoreV1(Dataset):
    r'''
        input z0: (FatJat basic stats, 1)
            'fj_pt',
            'fj_eta',
            'fj_sdmass',
            'fj_n_sdsubjets',
            'fj_doubleb',
            'fj_tau21',
            'fj_tau32',
            'npv',
            'npfcands',
            'ntracks',
            'nsv'

        input x0: (FatJet info, 1) (also input z1???)
        'fj_jetNTracks',
        'fj_nSV',
        'fj_tau0_trackEtaRel_0',
        'fj_tau0_trackEtaRel_1',
        'fj_tau0_trackEtaRel_2',
        'fj_tau1_trackEtaRel_0',
        'fj_tau1_trackEtaRel_1',
        'fj_tau1_trackEtaRel_2',
        'fj_tau_flightDistance2dSig_0',
        'fj_tau_flightDistance2dSig_1',
        'fj_tau_vertexDeltaR_0',
        'fj_tau_vertexEnergyRatio_0',
        'fj_tau_vertexEnergyRatio_1',
        'fj_tau_vertexMass_0',
        'fj_tau_vertexMass_1',
        'fj_trackSip2dSigAboveBottom_0',
        'fj_trackSip2dSigAboveBottom_1',
        'fj_trackSip2dSigAboveCharm_0',
        'fj_trackSipdSig_0',
        'fj_trackSipdSig_0_0',
        'fj_trackSipdSig_0_1',
        'fj_trackSipdSig_1',
        'fj_trackSipdSig_1_0',
        'fj_trackSipdSig_1_1',
        'fj_trackSipdSig_2',
        'fj_trackSipdSig_3',
        'fj_z_ratio'
        
        input x1: (Tracks, max 60)
        'trackBTag_EtaRel',
        'trackBTag_PtRatio',
        'trackBTag_PParRatio',
        'trackBTag_Sip2dVal',
        'trackBTag_Sip2dSig',
        'trackBTag_Sip3dVal',
        'trackBTag_Sip3dSig',
        'trackBTag_JetDistVal'
        
        input x2: (SVs, max 5)
        'sv_d3d',
        'sv_d3dsig',
        
        # truth categories are QCD=0 / Hbb=1
    '''

    url = 'root://cmseos.fnal.gov//store/group/lpccoffea/lgray/old_format'

    def __init__(self, root, transform=None):
        super(DeepJetCoreV1, self).__init__(root, transform)
        
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
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.z')))
        return raw_files

    @property
    def processed_file_names(self):
        return []


    def get(self, idx):
        file_idx = np.searchsorted(self.strides, idx) - 1
        idx_in_file = idx - self.strides[max(0, file_idx)]
        if file_idx >= self.strides.size:
            raise Exception(f'{idx} is beyond the end of the event list {self.strides[-1]}')
        edge_index = torch.empty((2,0), dtype=torch.long)
        with h5py.File(self.raw_paths[file_idx]) as f:
            x_jet = np.squeeze(f['x0'][idx_in_file])
            Ntrack = int(x_jet[0])
            if Ntrack > 0:
                x_track = f['x1'][idx_in_file,:Ntrack,:]
            else:
                Ntrack = 1
                x_track = np.zeros((1,8), dtype=np.float32)
            
            Nsv = int(x_jet[1])
            if Nsv > 0:
                x_sv = f['x2'][idx_in_file,:Nsv,:]
            else:
                Nsv = 1
                x_sv = np.zeros((1,2), dtype=np.float32)

            # convert to torch
            x_jet = torch.from_numpy(x_jet[2:])[None]
            x_track = torch.from_numpy(x_track)
            x_sv = torch.from_numpy(x_sv)
            
            
            # convert to non-onehot categories
            y = torch.from_numpy(f['y0'][idx_in_file])
            y = torch.argmax(y)
            
            # "z0" is the basic jet observables pt, eta, phi
            # store this as the usual x
            x = torch.from_numpy(f['z0'][idx_in_file])
            x_jet_raw = torch.from_numpy(f['z1'][idx_in_file])

            return Data(x=x, edge_index=edge_index, y=y,
                        x_jet=x_jet, x_track=x_track, x_sv=x_sv,
                        x_jet_raw=x_jet_raw)
