import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv


class Net(nn.Module):
    def __init__(self, hidden_dim = 16, pfc_input_dim = 13, dropout = 0.5):
        super(Net, self).__init__()
        
        
        self.pfc_encode = nn.Sequential(
            nn.Linear(pfc_input_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim-1)
        )

        self.graph_conv = DynamicEdgeConv(nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.SiLU()),
                k=8, aggr='mean')
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(2*hidden_dim, hidden_dim)
        )

    def forward(self, x_pfc):
        x_pfc_enc = self.pfc_encode(x_pfc)
        # concat the truth vertex value
        x_pfc_enc = torch.cat([x_pfc_enc, x_pfc[:,-1:]], dim=1)
        feats = self.graph_conv(x_pfc_enc)
        feats = self.ffn(feats)
        enc = torch.cat([feats, x_pfc], dim=1)
        return enc
