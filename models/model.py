import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear


class Net(nn.Module):
    def __init__(self, isFCN=False):
        self.isFCN = isFCN

        super(Net, self).__init__()
        hidden_dim = 32
        self.vtx_encode = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        #self.glob_encode = nn.Sequential(
        #    nn.Linear(1, hidden_dim),
        #    nn.ELU(),
        #    nn.Linear(hidden_dim, hidden_dim),
        #    nn.ELU()
        #)

        if not self.isFCN:
            self.pfc_encode = nn.Sequential(
                nn.Linear(13, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

            self.conv = DynamicEdgeConv(
                nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim//2), nn.LeakyReLU()),
                k=64, aggr = 'mean'
            )

            self.conv2 = DynamicEdgeConv(
                nn=nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()),
                k=64, aggr = 'mean'
            )

            self.output = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 4),
                nn.LeakyReLU(),
                nn.Linear(4, 1)
            )
        else:
            self.pfc_encode = nn.Sequential(
                nn.Linear(13, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )


    def forward(self,
                x_pfc, x_vtx,
                batch_pfc, batch_vtx):
        batch = batch_pfc

        if not self.isFCN:
            x_pfc_enc = self.pfc_encode(x_pfc)
            x_vtx_enc = self.vtx_encode(x_vtx)
            #x_glob_enc = self.glob_encode(x_glob)
            # dropout layer
            x_pfc_enc = F.dropout(x_pfc_enc, p=0.5, training=self.training)
            # create a representation of PFs to clusters
            feats1 = self.conv(x=(x_pfc_enc, x_pfc_enc), batch=(batch_pfc, batch_pfc))
            # similarly a representation of PFs-clusters amalgam to PFs
            # dropout layer
            feats1 = F.dropout(feats1, p=0.2, training=self.training)
            # get charged PFs
            charged_idx = torch.nonzero(x_pfc[:,11] != 0).squeeze()
            # select charged PFs in feats1
            charged_feats1 = feats1[charged_idx, :]
            charged_batch = batch[charged_idx]
            # concat x_vtx_enc and charged_feats1
            combined_feats = torch.cat((x_vtx_enc, charged_feats1), dim=0)
            combined_batch = torch.cat((batch_vtx, charged_batch), dim=0)
            # feats2 = self.conv(x=(combined_feats, feats1), batch=(combined_batch, batch_pfc))
            feats2 = self.conv(x=(charged_feats1, feats1), batch=(charged_batch, batch_pfc))
            # feats2 = self.conv(x=(feats1, charged_feats1), batch=(batch_pfc, charged_batch))
            # now to global variables
            #feats3 = self.conv(x=(x_glob_enc, feats2), batch=(batch_pfc, batch_pfc))

            #out, batch = avg_pool_x(batch, feats2, batch)        
            #out = self.output(out)
            out = self.output(feats2)
            #out = self.output(x_pfc_enc)
        else:
            out = self.pfc_encode(x_pfc)

        return out, batch, feats1, x_vtx_enc
