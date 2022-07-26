import torch
from torch import nn
import torch.nn.functional as F
# from torch_geometric.nn.conv import DynamicEdgeConv
try:
    from Point_transformer_conv import PointTransformerConv
except:
    from models.Point_transformer_conv import PointTransformerConv
from torch_cluster import knn

class Net(nn.Module):
    def __init__(self, hidden_dim=32, extra_charged_features = 1, pfc_input_dim=15, vtx_classes = 1, dropout=0.3, positional_dim = 4):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.pfc_input_dim = pfc_input_dim
        self.dropout = dropout
        self.vtx_classes = vtx_classes
        self.extra_charged_features = extra_charged_features
        self.positional_dim = positional_dim
    
    # connect all the particles in the same batch
        self.ptc_start = PointTransformerConv(in_channels = self.pfc_input_dim, out_channels = self.hidden_dim, pos_nn=nn.Linear(self.positional_dim, self.hidden_dim))
        self.ptc_final = PointTransformerConv(in_channels = self.hidden_dim, out_channels = self.vtx_classes + 1, pos_nn=nn.Linear(self.positional_dim, self.vtx_classes + 1))
        self.ptc_only = PointTransformerConv(in_channels = self.pfc_input_dim, out_channels = self.vtx_classes + 1, pos_nn=nn.Linear(self.positional_dim, self.vtx_classes + 1))
    def get_param_dict(self):
        return {
            'hidden_dim': self.hidden_dim,
            'pfc_input_dim': self.pfc_input_dim,
            'dropout': self.dropout,
            'vtx_classes': self.vtx_classes,
            'extra_charged_features': self.extra_charged_features,
            'positional_dim' : self.positional_dim
        }
    
    def get_param_str(self):
        return '_'.join([str(k) + '_' + str(v) for k, v in self.get_param_dict().items()])


    def forward(self, x_pfc, edge_index):
        pos_x_pfc = x_pfc[:, :self.positional_dim]
        # feats_1 = self.ptc_start(x_pfc, pos_x_pfc, edge_index)
        # feats_final = self.ptc_final(feats_1, pos_x_pfc, edge_index)
        return self.ptc_only(x_pfc, pos_x_pfc, edge_index.flip(0))
        
