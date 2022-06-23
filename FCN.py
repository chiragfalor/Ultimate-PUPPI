import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
import time
import sklearn
import numpy as np
from tqdm import tqdm
import sys
#sys.path.append('/home/yfeng/UltimatePuppi/deepjet-geometric/')
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
import os

class Net(nn.Module):
    def __init__(self, hidden_dim=32):
        super(Net, self).__init__()

        self.pfc_encode = nn.Sequential(
                nn.Linear(12, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self,
                x_pfc, x_vtx,
                batch_pfc, batch_vtx):
        batch = batch_pfc
        out = self.pfc_encode(x_pfc)
        return out, batch


BATCHSIZE = 32
start_time = time.time()
print("Training...")
data_train = UPuppiV0("/work/submit/cfalor/upuppi/z_reg/train/")
data_test = UPuppiV0("/work/submit/cfalor/upuppi/z_reg/test/")
train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_vtx'])


#import utils

model_dir = '/work/submit/cfalor/upuppi/z_reg/models/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
print("Using device: ", device, torch.cuda.get_device_name(0))

upuppi = Net().to(device)
optimizer = torch.optim.Adam(upuppi.parameters(), lr=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

def train():
    upuppi.train()
    counter = 0
    total_loss = 0
    for data in tqdm(train_loader):
        counter += 1
        data = data.to(device)
        optimizer.zero_grad()
        out = upuppi(data.x_pfc[:, :12], data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        loss = nn.MSELoss()(out[0][:,0], data.y)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    upuppi.eval()
    total_loss = 0
    counter = 0
    for data in test_loader:
        counter += 1
        data = data.to(device)
        out = upuppi(data.x_pfc[:, :12], data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        loss = nn.MSELoss()(out[0][:,0], data.y)
        total_loss += loss.item()
    return total_loss / len(test_loader.dataset)


for epoch in range(1, 20):
    loss = train()
    scheduler.step()
    loss_test = test()

    print('Epoch {:02d}, Loss: {:.8f}, Val_loss: {:.8f}'.format(
        epoch, loss, loss_test))

    state_dicts = {'model':upuppi.state_dict(),
                   'opt':optimizer.state_dict(),
                   'lr':scheduler.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))

