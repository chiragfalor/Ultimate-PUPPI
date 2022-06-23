import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv


class Net(nn.Module):
    def __init__(self, hidden_dim=16, pfc_input_dim=13, dropout=0.3, k1 = 64, k2 = 16):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.pfc_input_dim = pfc_input_dim
        self.dropout = dropout
        self.vtx_encode = nn.Sequential(
            nn.Linear(4, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
        self.pfc_encode = nn.Sequential(
            nn.Linear(pfc_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)),
            k=k1, aggr = 'mean'
        )

        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*(hidden_dim+pfc_input_dim), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)),
            k=k2, aggr = 'mean')

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 4),
            nn.SiLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x_pfc, x_vtx, batch_pfc, batch_vtx):
        batch = batch_pfc
        x_pfc_enc = self.pfc_encode(x_pfc)
        x_vtx_enc = self.vtx_encode(x_vtx)

        x_pfc_enc = F.dropout(x_pfc_enc, p=self.dropout, training=self.training)
        # create a representation of PFs to clusters
        feats1 = self.conv(x=(x_pfc_enc, x_pfc_enc), batch=(batch_pfc, batch_pfc))
        concat_feats = torch.cat([x_pfc, feats1], dim=1)
        # dropout layer
        concat_feats = F.dropout(concat_feats, p=self.dropout, training=self.training)
        # get charged PFs
        charged_idx = torch.nonzero(x_pfc[:,11] != 0).squeeze()
        # select charged PFs in feats1
        charged_feats1 = concat_feats[charged_idx]
        charged_batch = batch[charged_idx]
        # concat x_vtx_enc and charged_feats1
        # combined_feats = torch.cat((x_vtx_enc, charged_feats1), dim=0)
        # combined_batch = torch.cat((batch_vtx, charged_batch), dim=0)
        # feats2 = self.conv(x=(combined_feats, feats1), batch=(combined_batch, batch_pfc))
        feats2 = self.conv2(x=(charged_feats1, concat_feats), batch=(charged_batch, batch_pfc))
        
        # pass the features to the dense output layer
        out = self.output(feats2)

        return out, batch, feats1, x_vtx_enc
