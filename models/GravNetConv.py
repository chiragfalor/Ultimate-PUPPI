import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GravNetConv
from torch_geometric.nn.pool import avg_pool_x


class Net(nn.Module):
    def __init__(self, hidden_dim=16, pfc_input_dim=13, dropout=0.2):
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

        self.conv1 = GravNetConv(in_channels=hidden_dim, out_channels=hidden_dim, space_dimensions = hidden_dim//2, propagate_dimensions = 8, k=128)
        self.conv2 = GravNetConv(in_channels=hidden_dim+pfc_input_dim, out_channels=hidden_dim, space_dimensions = hidden_dim//2, propagate_dimensions = 8, k=128)


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
        feats1 = self.conv1(x=x_pfc_enc, batch=batch_pfc)
        # add layer normalization
        feats1 = F.layer_norm(feats1, [feats1.shape[1]])
        # dropout layer
        feats1 = F.dropout(feats1, p=self.dropout, training=self.training)
        # concat pfc and feats1
        pfc_feats1 = torch.cat([x_pfc, feats1], dim=1)
        feats2 = self.conv2(x=pfc_feats1, batch=batch_pfc)
        # add layer normalization
        feats2 = F.layer_norm(feats2, [feats2.shape[1]])\
        # dropout layer
        feats2 = F.dropout(feats2, p=self.dropout, training=self.training)
        feats = torch.cat([feats1, feats2], dim=1)
        # concat the last feature of pfc
        feats = self.ffn(feats)
        feats = torch.cat([feats, x_pfc[:,:-1]], dim=1)
        out = self.output(feats)
        return out, batch, x_pfc_enc
