import time
import sklearn
import numpy as np
import sys
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
import os
import torch
from torch import nn
from tqdm import tqdm
from save_results import save_predictions

BATCHSIZE = 32
NUM_EPOCHS = 5
start_time = time.time()
data_train = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/train/")
data_test = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/test/")

train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_vtx'])


# model = "DynamicGCN"
# model = "GAT"
model = "GravNetConv"
# model = "No_Encode_grav_net"

# import DynamicGCN.py or GAT.py in models folder
if model == "DynamicGCN":
    from models.DynamicGCN import Net
elif model == "GAT":
    from models.GAT import Net
elif model == "GravNetConv":
    from models.GravNetConv import Net
elif model == "No_Encode_grav_net":
    from models.No_Encode_grav_net import Net
else:
    raise(Exception("Model not found"))

#import utils

model_dir = '/work/submit/cfalor/upuppi/deepjet-geometric/models/{}/'.format(model)

print("Training {}...".format(model))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device, torch.cuda.get_device_name(0))

# hidden_dim = 16
upuppi = Net().to(device)
optimizer = torch.optim.Adam(upuppi.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

def train(k=5):
    # k is the ratio of loss weight of neutral particles to charged particles
    upuppi.train()
    counter = 0
    total_loss = 0
    for data in tqdm(train_loader):
        counter += 1
        data = data.to(device)
        optimizer.zero_grad()
        out = upuppi(data.x_pfc, data.x_pfc_batch)

        # calculate neutral loss
        neutral_indices = torch.nonzero(data.x_pfc[:, 11] == 0).squeeze()
        neutral_out = out[0][:,0][neutral_indices]
        neutral_y = data.y[neutral_indices]
        neutral_loss = nn.MSELoss()(neutral_out, neutral_y)
        # calculate charged loss
        charged_indices = torch.nonzero(data.x_pfc[:,11] != 0).squeeze()
        charged_out = out[0][:,0][charged_indices]
        charged_y = data.y[charged_indices]
        charged_loss = nn.MSELoss()(charged_out, charged_y)
        # calculate total loss
        loss = (k*neutral_loss + charged_loss)/(k+1)
        # loss = nn.MSELoss()(out[0][:,0], data.y)
        # loss = nn.L1Loss()(out[0][:,0], data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test():
    upuppi.eval()
    total_loss = 0
    counter = 0
    for data in test_loader:
        counter += 1
        data = data.to(device)
        out = upuppi(data.x_pfc,data.x_pfc_batch)
        loss = nn.MSELoss()(out[0][:,0], data.y)
        total_loss += loss.item()
    return total_loss / len(test_loader.dataset)


for epoch in range(1, NUM_EPOCHS + 1):
    loss = train(epoch*2)
    scheduler.step()
    test_loss = test()

    print('Epoch {:02d}, Loss: {:.8f}, Test_loss: {:.8f}'.format(
        epoch, loss, test_loss))

    state_dicts = {'model':upuppi.state_dict(),
                   'opt':optimizer.state_dict(),
                   'lr':scheduler.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))




print("Training took", time.time() - start_time, "to run")
print("Saving predictions...")
save_predictions(upuppi, test_loader, model)
