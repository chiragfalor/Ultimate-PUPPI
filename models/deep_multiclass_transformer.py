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
    def __init__(self, hidden_dim=160, extra_charged_features = 1, pfc_input_dim=12, vtx_classes = 1, dropout=0.3, k1 = 32, k2 = 16):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.pfc_input_dim = pfc_input_dim
        self.dropout = dropout
        self.vtx_classes = vtx_classes
        self.extra_charged_features = extra_charged_features
        self.k1 = k1
        self.k2 = k2


        self.vtx_encode_1 = nn.Sequential(
            nn.Linear(5, hidden_dim//4),
            nn.SiLU(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.SiLU(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.neutral_pfc_encode = nn.Sequential(
            nn.Linear(pfc_input_dim - extra_charged_features, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.Dropout(p=dropout, inplace=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.charged_pfc_encode = nn.Sequential(
            nn.Linear(pfc_input_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.Dropout(p=dropout, inplace=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )


        
        self.charged_pfc_encode_2 = nn.Sequential(nn.Linear(pfc_input_dim, hidden_dim))
        self.neutral_pfc_encode_2 = nn.Sequential(nn.Linear(pfc_input_dim - extra_charged_features, hidden_dim))



        self.vtx_encode_2 = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.SiLU(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.Dropout(p=dropout, inplace=True),
            nn.SiLU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

        self.pfc_encode_2 = nn.Sequential(
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.SiLU(),
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )
    def conv1(self, x, pos, batch):
            # x, pos, batch are dual tensors
        ptc1 = PointTransformerConv(in_channels = self.hidden_dim, 
                                    out_channels = self.hidden_dim, 
                                    pos_nn = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim)), 
                                    attn_nn=nn.Linear(self.hidden_dim, self.hidden_dim)
                                        ).to(x[0].device)
        edge_index = knn(pos[0], pos[1], self.k1, batch[0], batch[1]).flip([0])
        # print device of all tensors
        # print(x.device, pos.device, batch.device, edge_index.device)
        return ptc1(x, pos, edge_index)

    def conv2(self, x, pos, batch):
        # x, pos, batch are dual tensors
        ptc2 = PointTransformerConv(in_channels = self.hidden_dim, 
                                    out_channels = self.hidden_dim,
                                    pos_nn = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim)), 
                                    attn_nn=nn.Linear(self.hidden_dim, self.hidden_dim)
                                        ).to(x[0].device)
        edge_index = knn(pos[0], pos[1], self.k2, batch[0], batch[1]).flip([0])
        return ptc2(x, pos, edge_index)

    def get_param_dict(self):
        return {
            'hidden_dim': self.hidden_dim,
            'pfc_input_dim': self.pfc_input_dim,
            'dropout': self.dropout,
            'vtx_classes': self.vtx_classes,
            'extra_charged_features': self.extra_charged_features,
            'k1': self.k1,
            'k2': self.k2
        }
    
    def score_from_encodings(self, pfc_enc, target_vector):
        return torch.sum(pfc_enc*target_vector, dim=1)

    def vtx_scores_from_encodings(self, pfc_enc, target_vertices):
        pfc_scores_per_vtx = torch.zeros(pfc_enc.shape[0], target_vertices.shape[0] + 1).to(pfc_enc.device)
        for i in range(target_vertices.shape[0]):
            pfc_scores_per_vtx[:, i] = torch.sum(pfc_enc*target_vertices[i], dim=1)
        # the score of the last column is the -sum of the scores of the other columns
        pfc_scores_per_vtx[:, -1] = -torch.sum(pfc_scores_per_vtx[:, :-1], dim=1)
        return pfc_scores_per_vtx

    def pfc_scores_from_top_vtxs(self, pfc_enc, vtx_enc, pfc_batch, vtx_batch):
        '''
        returns the scores of the PFs by dotting the PF encodings with the top vertices
        returns a tensor of shape (num_pfcs, vtx_classes)
        '''
        # get batch size, maximum of the batch indices
        batch_size = torch.max(pfc_batch) + 1
        for i in range(batch_size):
            # select the top self.vtx_classes vertices for each batch
            target_vertices = vtx_enc[vtx_batch == i, :][:self.vtx_classes]
            if target_vertices.shape[0] < self.vtx_classes:
                raise(ValueError('Not enough vertices in batch'))
            # get the scores of the PFs for the batch
            pfc_scores = self.vtx_scores_from_encodings(pfc_enc[pfc_batch == i, :], target_vertices)
            # add the scores to the batch scores
            if i == 0:
                pfc_scores_batch = pfc_scores
            else:
                pfc_scores_batch = torch.cat((pfc_scores_batch, pfc_scores), dim=0)
        return pfc_scores_batch



    def forward(self, x_pfc, x_vtx, batch_pfc, batch_vtx):

        x_vtx_euc_enc = self.vtx_encode_1(x_vtx)

        charged_mask = (x_pfc[:, -2] != 0).type(torch.float).unsqueeze(1)
        neutral_mask = 1 - charged_mask
        
        # encode charged and neutral with different encoders
        charged_pfc_enc = self.charged_pfc_encode(x_pfc)*charged_mask
        neutral_pfc_enc = self.neutral_pfc_encode(x_pfc[:, :-self.extra_charged_features])*neutral_mask

        x_pfc_init_enc = charged_pfc_enc + neutral_pfc_enc

        x_pfc_init_enc = F.dropout(x_pfc_init_enc, p=self.dropout, training=self.training)

        # create a representation of PFs to clusters
        x_pfc_euc_enc = self.conv1((x_pfc_init_enc, x_pfc_init_enc), (x_pfc_init_enc, x_pfc_init_enc), (batch_pfc, batch_pfc))
        x_pfc_euc_enc = self.conv1((x_pfc_euc_enc, x_pfc_euc_enc), (x_pfc_euc_enc, x_pfc_euc_enc), (batch_pfc, batch_pfc))
        x_pfc_euc_enc = F.dropout(x_pfc_euc_enc, p=self.dropout, training=self.training)
        
        # concat_feats = torch.cat([x_pfc[:, :-1], x_pfc_euc_enc], dim=1)
        orig_feats = self.charged_pfc_encode_2(x_pfc)*charged_mask + self.neutral_pfc_encode_2(x_pfc[:,:-self.extra_charged_features])*neutral_mask
        charged_mask = (x_pfc[:, -2] != 0)
        charged_orig_feats, charged_pos_enc, charged_batch = orig_feats[charged_mask], x_pfc_euc_enc[charged_mask], batch_pfc[charged_mask]
        
        feats2 = self.conv2(x=(charged_orig_feats, orig_feats), pos=(charged_pos_enc, x_pfc_euc_enc), batch=(charged_batch, batch_pfc))
        charged_feats2 = feats2[charged_mask]
        feats2 = self.conv2(x=(charged_feats2, feats2), pos=(charged_feats2, feats2), batch=(charged_batch, batch_pfc))
        feats2 = F.dropout(feats2, p=self.dropout, training=self.training)

        x_vtx_final_enc = self.vtx_encode_2(x_vtx_euc_enc)
        x_pfc_final_enc = self.pfc_encode_2(feats2)
        scores = self.pfc_scores_from_top_vtxs(x_pfc_final_enc, x_vtx_final_enc, batch_pfc, batch_vtx)

        return scores, x_pfc_euc_enc, x_vtx_euc_enc
