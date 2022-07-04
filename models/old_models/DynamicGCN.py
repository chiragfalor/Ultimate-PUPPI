import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear


class Net(nn.Module):
    def __init__(self, hidden_dim=16, pfc_input_dim=13, dropout=0.8):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.pfc_input_dim = pfc_input_dim
        self.pfc_encode = nn.Sequential(
                nn.Linear(self.pfc_input_dim, hidden_dim//2),
                nn.ELU(),
                nn.Linear(hidden_dim//2, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv = DynamicEdgeConv(
                nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
                k=100, aggr='mean'
            )


        self.ffn = nn.Sequential(
                nn.Linear(2*hidden_dim, 16),
                nn.ELU(),
                nn.Linear(16, 4)
            )

        self.output = nn.Sequential(
                nn.Linear(4+self.pfc_input_dim-1, 4),
                nn.ELU(),
                nn.Linear(4, 1)
            )



    def forward(self,x_pfc, batch_pfc):
        batch = batch_pfc
        x_pfc_enc = self.pfc_encode(x_pfc)
        feats1 = self.conv(x=(x_pfc_enc, x_pfc_enc), batch=(batch_pfc, batch_pfc))
        # add layer normalization
        feats1 = F.layer_norm(feats1, [feats1.shape[1]])
        # add dropout layer
        feats1 = F.dropout(feats1, p=self.dropout, training=self.training)
        feats2 = self.conv(x=(x_pfc_enc, feats1), batch=(batch_pfc, batch_pfc))
        # add layer normalization
        feats2 = F.layer_norm(feats2, [feats2.shape[1]])
        # dropout layer
        feats2 = F.dropout(feats2, p=self.dropout, training=self.training)
        feats = torch.cat([feats1, feats2], dim=1)
        # concat the last feature of pfc
        feats = self.ffn(feats)
        feats = torch.cat([feats, x_pfc[:, :-1]], dim=1)
        out = self.output(feats)
        return out, batch
