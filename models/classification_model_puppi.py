import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv


class Net(nn.Module):
    def __init__(self, hidden_dim=160, pfc_input_dim=12, dropout=0.3, k1 = 32, k2 = 16, aggr = 'mean'):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.pfc_input_dim = pfc_input_dim
        self.dropout = dropout


        self.vtx_encode = nn.Sequential(
            nn.Linear(5, hidden_dim//4),
            nn.SiLU(),
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

        self.neutral_pfc_encode = nn.Sequential(
            nn.Linear(pfc_input_dim - 1, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

        self.charged_pfc_encode = nn.Sequential(
            nn.Linear(pfc_input_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)),
            k=k1, aggr = aggr
        )

        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*(hidden_dim+pfc_input_dim-1), hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)),
            k=k2, aggr = aggr)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.SiLU(),
            nn.Linear(hidden_dim//4, 4),
            nn.SiLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x_pfc, x_vtx, batch_pfc, batch_vtx):

        x_vtx_enc = self.vtx_encode(x_vtx)

        charged_mask = (x_pfc[:, -2] != 0).type(torch.float).unsqueeze(1)
        neutral_mask = 1 - charged_mask
        
        # encode charged and neutral with different encoders
        charged_pfc_enc = self.charged_pfc_encode(x_pfc)*charged_mask
        neutral_pfc_enc = self.neutral_pfc_encode(x_pfc[:, :-1])*neutral_mask

        x_pfc_enc = charged_pfc_enc + neutral_pfc_enc

        x_pfc_enc = F.dropout(x_pfc_enc, p=self.dropout, training=self.training)


        # create a representation of PFs to clusters
        feats1 = self.conv(x=(x_pfc_enc, x_pfc_enc), batch=(batch_pfc, batch_pfc))
        concat_feats = torch.cat([x_pfc[:, :-1], feats1], dim=1)
        concat_feats = F.dropout(concat_feats, p=self.dropout, training=self.training)
        

        charged_mask = (x_pfc[:, -2] != 0)
        charged_concat_feats, charged_batch = concat_feats[charged_mask], batch_pfc[charged_mask]
        
        feats2 = self.conv2(x=(charged_concat_feats, concat_feats), batch=(charged_batch, batch_pfc))
        
        # pass the features to the dense output layer
        out = self.output(feats2)

        return out, batch_pfc, feats1, x_vtx_enc
