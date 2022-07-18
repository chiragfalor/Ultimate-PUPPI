import torch
from torch import nn
import torch.nn.functional as F
# from torch_geometric.nn.conv import Point_Transformer_Conv
try:
    from Point_transformer_conv import PointTransformerConv
except:
    from models.Point_transformer_conv import PointTransformerConv
from torch_cluster import knn

class Net(nn.Module):
    def __init__(self, hidden_dim=32, pfc_input_dim=13, dropout=0.3, k1 = 64, k2 = 16):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.pfc_input_dim = pfc_input_dim
        self.dropout = dropout
        self.k1 = k1
        self.k2 = k2
        self.vtx_encode = nn.Sequential(
            nn.Linear(5, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pfc_encode = nn.Sequential(
            nn.Linear(pfc_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = PointTransformerConv(in_channels = hidden_dim, out_channels = hidden_dim, pos_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = PointTransformerConv(in_channels = pfc_input_dim, out_channels = hidden_dim, pos_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)))
        
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
        
        pfc_position_encoding = x_pfc_enc
        edge_index = knn(pfc_position_encoding, pfc_position_encoding, self.k1, batch, batch).flip([0])
        feats1 = self.conv1(x = pfc_position_encoding, pos = pfc_position_encoding, edge_index = edge_index)
        feats1 = F.dropout(feats1, p=self.dropout, training=self.training)
        
        # second convolution layer
        charged_idx = torch.nonzero(x_pfc[:,-2] != 0).squeeze()
        charged_feats1 = feats1[charged_idx]
        charged_batch = batch[charged_idx]
        charged_x_pfc = x_pfc[charged_idx]
        edge_index = knn(charged_feats1, feats1, self.k2, charged_batch, batch).flip([0])
        feats2 = self.conv2(x = (charged_x_pfc, x_pfc), pos = (charged_feats1, feats1), edge_index = edge_index)
        
        # pass the features to the dense output layer
        out = self.output(feats2)

        return out, batch, feats1, x_vtx_enc
