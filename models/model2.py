import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear


class Net(nn.Module):
    def __init__(self, hidden_dim = 16):
        super(Net, self).__init__()
        self.vtx_encode = nn.Sequential(
            nn.Linear(4, hidden_dim//4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
        self.pfc_encode = nn.Sequential(
            nn.Linear(13, hidden_dim//2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

            # self.conv = DynamicEdgeConv(
            #     nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.LeakyReLU()),
            #     k=20
            # )

            # self.output = nn.Sequential(
            #     nn.Linear(hidden_dim, 64),
            #     nn.LeakyReLU(),
            #     nn.Linear(64, 32),
            #     nn.LeakyReLU(),
            #     nn.Linear(32, 4),
            #     nn.LeakyReLU(),
            #     nn.Linear(4, 1), nn.LeakyReLU()
            # )


    def forward(self,
                x_pfc, x_vtx,
                batch_pfc, batch_vtx):
        x_pfc_enc = self.pfc_encode(x_pfc)
        x_vtx_enc = self.vtx_encode(x_vtx)
        return x_pfc_enc, x_vtx_enc
