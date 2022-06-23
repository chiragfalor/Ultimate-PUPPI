from deepjet_geometric.datasets import DeepJetCoreV1
from torch_geometric.data import DataLoader

import os

test = DeepJetCoreV1(os.path.join(os.getcwd(), 'dummy_data'))

load_train = DataLoader(test, batch_size=20, shuffle=True,
                        follow_batch=['x_track', 'x_sv'])

import torch
from torch import nn
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_dim = 16
        
        self.sv_encode = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.trk_encode = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim)),
            k=8
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_sv, x_trk, batch_sv, batch_trk):
        x_sv_enc = self.sv_encode(x_sv)
        x_trk_enc = self.trk_encode(x_trk)
                
        feats = self.conv(x=(x_sv_enc, x_trk_enc), batch=(batch_sv, batch_trk))
        
        out = self.output(feats)
        batch = batch_trk
        
        out, batch = avg_pool_x(batch, out, batch)
        
        return out, batch
        
dummy = Net()
dummy.eval() # just for example

for data in load_train:
    out = dummy(data.x_sv,
                data.x_track,
                data.x_sv_batch,
                data.x_track_batch)
                
    print(data.x_sv.size(), data.x_track.size(), out[0].size())
    print(out[0])
    print(out[1])
    #print(data)
