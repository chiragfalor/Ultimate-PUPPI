import sklearn
import numpy as np

from deepjet_geometric.datasets import MetV1
from torch_geometric.data import DataLoader
import os

BATCHSIZE = 30

data_train = MetV1("/data/t3home000/bmaier/tor_met/train_v2/")
data_test = MetV1("/data/t3home000/bmaier/tor_met/test_v2/")

train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_clus', 'x_glob'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_clus', 'x_glob'])

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear

import utils

OUTPUT = '/home/bmaier/public_html/figs/puma/geometric_v2/'
model_dir = '/data/t3home000/bmaier/puma/geometric_v2/'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_dim = 128
        
        self.clus_encode = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        
        #self.glob_encode = nn.Sequential(
        #    nn.Linear(1, hidden_dim),
        #    nn.ELU(),
        #    nn.Linear(hidden_dim, hidden_dim),
        #    nn.ELU()
        #)
        
        self.pfc_encode = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=8
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 4),
            nn.ELU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                x_pfc, x_clus, x_glob,
                batch_pfc, batch_clus, batch_glob):
        x_pfc_enc = self.pfc_encode(x_pfc)
        x_clus_enc = self.clus_encode(x_clus)
        #x_glob_enc = self.glob_encode(x_glob)
        
        # create a representation of PFs to clusters
        feats1 = self.conv(x=(x_clus_enc, x_pfc_enc), batch=(batch_clus, batch_pfc))
        # similarly a representation of PFs-clusters amalgam to PFs
        feats2 = self.conv(x=(feats1, x_pfc_enc), batch=(batch_pfc, batch_pfc))
        # now to global variables
        #feats3 = self.conv(x=(x_glob_enc, feats2), batch=(batch_pfc, batch_pfc))

        batch = batch_pfc
        #out, batch = avg_pool_x(batch, feats2, batch)        
        #out = self.output(out)
        out = self.output(feats2)
        
        return out, batch
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

puma = Net().to(device)
#puma.load_state_dict(torch.load(model_dir+"epoch-32.pt")['model'])
optimizer = torch.optim.Adam(puma.parameters(), lr=0.001)
#optimizer.load_state_dict(torch.load(model_dir+"epoch-32.pt")['opt'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#scheduler.load_state_dict(torch.load(model_dir+"epoch-32.pt")['lr'])

def train():
    puma.train()
    counter = 0

    total_loss = 0
    for data in train_loader:
        counter += 1
        print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)))
        data = data.to(device)
        optimizer.zero_grad()
        out = puma(data.x_pfc,
                    data.x_clus,
                    data.x_glob,
                    data.x_pfc_batch,
                    data.x_clus_batch,
                    data.x_glob_batch)

        loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1)[data.y>-1.],data.y[data.y>-1.].float())
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(train_loader.dataset)


for epoch in range(1, 50):
    loss = train()
    scheduler.step()

    print('Epoch {:03d}, Loss: {:.8f}'.format(
        epoch, loss))

    state_dicts = {'model':puma.state_dict(),
                   'opt':optimizer.state_dict(),
                   'lr':scheduler.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))

