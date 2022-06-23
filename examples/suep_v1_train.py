import sklearn
import numpy as np
from random import randrange
from Disco import distance_corr

from deepjet_geometric.datasets import SUEPV1
#from deepjet_geometric.networks import SuepNet

from torch_geometric.data import DataLoader

import os
import argparse

BATCHSIZE = 512

parser = argparse.ArgumentParser(description='Test.')
parser.add_argument('--ipath', action='store', type=str, help='Path to input files.')
parser.add_argument('--vpath', action='store', type=str, help='Path to validation files.')
parser.add_argument('--opath', action='store', type=str, help='Path to save models and plots.')
args = parser.parse_args()

print(args.ipath)

data_train = SUEPV1(args.ipath)
data_test = SUEPV1(args.vpath)

train_loader = DataLoader(data_train, batch_size=BATCHSIZE,shuffle=True,
                          follow_batch=['x_pf'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE,shuffle=True,
                         follow_batch=['x_pf'])

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
from torch_geometric.nn import DataParallel

model_dir = args.opath


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        hidden_dim = 16
        
        self.pf_encode = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        

        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.ELU()),
            k=8
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ELU(),
            nn.Linear(8, 4),
            nn.ELU(),
            nn.Linear(4, 2)
            #nn.Sigmoid()    
            )
        
    def forward(self,
                x_pf,
                batch_pf):

        x_pf_enc = self.pf_encode(x_pf)
        
        # create a representation of PFs to PFs
        feats1 = self.conv(x=(x_pf_enc, x_pf_enc), batch=(batch_pf, batch_pf))

        batch = batch_pf
        out, batch = avg_pool_x(batch, feats1, batch)
        
        out = self.output(out)
        
        return out, batch
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#model = DataParallel(model)

suep = Net().to(device)
#suep.load_state_dict(torch.load(model_dir+"epoch-32.pt")['model'])
optimizer = torch.optim.Adam(suep.parameters(), lr=0.001)
#optimizer = torch.optim.Adam(disco.parameters(), lr=0.001)
#optimizer.load_state_dict(torch.load(model_dir+"epoch-32.pt")['opt'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#scheduler.load_state_dict(torch.load(model_dir+"epoch-32.pt")['lr'])


def train():
    suep.train()
    counter = 0

    total_loss = 0
    for data in train_loader:
        counter += 1

        print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)),end='\r')
        data = data.to(device)
        optimizer.zero_grad()
        out = suep(data.x_pf,
                    data.x_pf_batch)

        # ABCDisco loss start
        loss1 = nn.BCEWithLogitsLoss()(torch.squeeze(out[0][:,0]).view(-1),data.y.float())
        loss2 = nn.BCEWithLogitsLoss()(torch.squeeze(out[0][:,1]).view(-1),data.y.float())
        
        bkgnn1 = out[0][:,0]
        bkgnn1 = bkgnn1[(data.y==0)]
        bkgnn2 = out[0][:,1]
        bkgnn2 = bkgnn2[(data.y==0)]
        lambdaval = 1000
        loss = loss1 + loss2 + lambdaval*distance_corr(bkgnn1,bkgnn2)
        # ABCDisco loss end

        loss.backward()
        total_loss += loss.item()

        optimizer.step()

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    suep.eval()
    total_loss = 0
    counter = 0
    for data in test_loader:
        counter += 1
        print(str(counter*BATCHSIZE)+' / '+str(len(test_loader.dataset)),end='\r')
        data = data.to(device)
        with torch.no_grad():
            out = suep(data.x_pf,
                       data.x_pf_batch)

            loss1 = nn.BCEWithLogitsLoss()(torch.squeeze(out[0][:,0]).view(-1),data.y.float())
            loss2 = nn.BCEWithLogitsLoss()(torch.squeeze(out[0][:,1]).view(-1),data.y.float())

            bkgnn1 = out[0][:,0]
            bkgnn1 = bkgnn1[(data.y==0)]
            bkgnn2 = out[0][:,1]
            bkgnn2 = bkgnn2[(data.y==0)]
            lambdaval = 1000
            loss = loss1 + loss2 + lambdaval*distance_corr(bkgnn1,bkgnn2) 
            
            total_loss += loss.item()

    return total_loss / len(test_loader.dataset)

for epoch in range(1, 50):
    loss = train()
    scheduler.step()

    loss_val = test()

    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
        epoch, loss, loss_val))

    #print('Epoch {:03d}, Loss: {:.8f}'.format(
    #    epoch, loss))


    state_dicts = {'model':suep.state_dict(),
                   'opt':optimizer.state_dict(),
                   'lr':scheduler.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))


