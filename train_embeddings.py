import time
import sklearn
import numpy as np
import sys
#sys.path.append('/home/yfeng/UltimatePuppi/deepjet-geometric/')
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
import os
import torch
from torch import nn
from models.model2 import Net
from tqdm import tqdm

# load the home directory path
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()


BATCHSIZE = 32
start_time = time.time()
print("Training...")
data_train = UPuppiV0(home_dir + 'train/')
data_test = UPuppiV0(home_dir + 'test/')


train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_vtx'])

model = "embedding_model"
model_dir = home_dir + 'models/{}/'.format(model)
#model_dir = '/home/yfeng/UltimatePuppi/deepjet-geometric/models/v0/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# print the device used
print("Using device: ", device, torch.cuda.get_device_name(0))

# create the model
upuppi = Net().to(device)
optimizer = torch.optim.Adam(upuppi.parameters(), lr=0.001)

def loss_fn(data, pfc_enc, vtx_enc):
    total_pfc_loss = 0
    total_vtx_loss = 0
    reg_loss = 0
    euclidean_loss = nn.MSELoss().to(device)
    batch_size = data.x_pfc_batch.max().item() + 1
    for i in range(batch_size):
        # get the batch index of the current batch
        pfc_indices = (data.x_pfc_batch == i)
        vtx_indices = (data.x_vtx_batch == i)
        # get the embedding of the pfc, vtx, and truth in the current batch
        pfc_enc_batch = pfc_enc[pfc_indices, :]
        vtx_enc_batch = vtx_enc[vtx_indices, :]
        truth_batch = data.truth[pfc_indices].to(dtype=torch.int64, device=device)
        # take out particles which have corresponding vertices
        valid_pfc = (truth_batch >= 0)
        truth_batch = truth_batch[valid_pfc]
        pfc_enc_batch = pfc_enc_batch[valid_pfc, :]
        # the true encoding is the embedding of the true vertex
        vertex_encoding = vtx_enc_batch[truth_batch, :]
        # calculate loss between pfc encoding and vertex encoding
        pfc_loss = 0.5*euclidean_loss(pfc_enc_batch, vertex_encoding)
        total_pfc_loss += pfc_loss

        # now regularize to keep vertices far
        random_indices = torch.randperm(len(truth_batch))[:30]
        random_vtx_encoding = vertex_encoding[random_indices, :]
        for j in range(len(random_vtx_encoding)):
            for k in range(j+1, len(random_vtx_encoding)):
                vtx_loss = -0.001*euclidean_loss(random_vtx_encoding[j, :], random_vtx_encoding[k, :])
                total_vtx_loss += vtx_loss
        
        # regularize the whole embedding to keep it normalized
    reg_loss = ((torch.norm(vtx_enc, p=2, dim=1)/10)**6).mean()
    # print the losses
    print("Pfc loss: ", total_pfc_loss.item(), " Vtx loss: ", total_vtx_loss.item(), " Reg loss: ", reg_loss.item())
    return total_pfc_loss + total_vtx_loss + reg_loss





def train():
    upuppi.train()
    counter = 0
    total_loss = 0
    for data in tqdm(train_loader):
        counter += 1
        data = data.to(device)
        optimizer.zero_grad()
        pfc_enc, vtx_enc = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        loss = loss_fn(data, pfc_enc, vtx_enc)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if counter % 100 == 0:
            print("Iteration: ", counter, " Loss: ", total_loss)
    total_loss = total_loss / counter        
    return total_loss

# test function
@torch.no_grad()
def test():
    upuppi.eval()
    euclidean_loss = nn.MSELoss()
    counter = 0
    total_loss = 0
    for data in tqdm(test_loader):
        counter += 1
        data = data.to(device)
        optimizer.zero_grad()
        pfc_enc, vtx_enc = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        loss = loss_fn(data, pfc_enc, vtx_enc)
        total_loss += loss.item()
    total_loss = total_loss / counter        
    return total_loss

# train the model

for epoch in range(10):
    loss = 0
    test_loss = 0
    loss = train()
    state_dicts = {'model':upuppi.state_dict(),
                   'opt':optimizer.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
    print("Model saved")
    print("Time elapsed: ", time.time() - start_time)
    print("-----------------------------------------------------")
    test_loss = test()
    print("Epoch: ", epoch, " Loss: ", loss, " Test Loss: ", test_loss)

    

