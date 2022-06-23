from deepjet_geometric.datasets import DeepJetCoreV2
from torch_geometric.data import DataLoader

import os

test = DeepJetCoreV2(os.path.join(os.getcwd(), 'dummy_data2'))

load_train = DataLoader(test, batch_size=20, shuffle=True,
                        follow_batch=['x_track', 'x_sv', 'x_pfc'])

import torch
from torch import nn
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_dim = 32
        
        self.sv_encode = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        self.trk_encode = nn.Sequential(
            nn.Linear(30, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        self.pfc_encode = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        self.conv_trk = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=8
        )
        
        self.conv_pfc = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=8
        )
        
        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=8
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                x_sv, x_trk, x_pfc,
                batch_sv, batch_trk, batch_pfc):
        x_sv_enc = self.sv_encode(x_sv)
        x_trk_enc = self.trk_encode(x_trk)
        x_pfc_enc = self.pfc_encode(x_pfc)
        
        # create a representation of svs to tracks
        feats1 = self.conv(x=(x_sv_enc, x_trk_enc), batch=(batch_sv, batch_trk))
        # similarly pfcandidates to tracks
        feats2 = self.conv(x=(x_pfc_enc, x_trk_enc), batch=(batch_pfc, batch_trk))
        # now compare the SV - track amalgam to the PFCand - track one
        feats3 = self.conv(x=(feats1, feats2), batch=(batch_trk, batch_trk))
        
        batch = batch_trk
        out, batch = avg_pool_x(batch, feats3, batch)
        
        out = self.output(out)
        
        return out, batch
        
dummy = Net()
dummy.eval() # just for example

for data in load_train:
    out = dummy(data.x_sv,
                data.x_track,
                data.x_pfc,
                data.x_sv_batch,
                data.x_track_batch,
                data.x_pfc_batch)
                
    print(data.x_sv.size(), data.x_track.size(), out[0].size())
    print(out[0])
    print(out[1])
    #print(data)
