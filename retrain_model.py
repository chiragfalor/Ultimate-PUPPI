# pyright: reportMissingModuleSource=false

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
from models.modelv2 import Net
from tqdm import tqdm

# load the home directory path
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()

BATCHSIZE = 64
start_time = time.time()
data_train = UPuppiV0(home_dir + 'train/')
data_test = UPuppiV0(home_dir + 'test/')

# data_train = UPuppiV0(home_dir + 'train2/')
# data_test = UPuppiV0(home_dir + 'test2/')


train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_vtx'])

model = "combined_model"
model = "Dynamic_GATv2"
model = "modelv2"
# model = "modelv2_neg"
# model = "modelv2_nz199"
# model = "modelv2_nz0"
# model = "modelv3"
model_dir = home_dir + 'models/{}/'.format(model)
#model_dir = '/home/yfeng/UltimatePuppi/deepjet-geometric/models/v0/'


print("Training {}...".format(model))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# print the device used
print("Using device: ", device, torch.cuda.get_device_name(0))

# create the model

epoch_to_load = 40
upuppi = Net(pfc_input_dim=13).to(device)
optimizer = torch.optim.Adam(upuppi.parameters(), lr=0.001)
model_dir = home_dir + 'models/{}/'.format(model)
model_loc = os.path.join(model_dir, 'epoch-{}.pt'.format(epoch_to_load))
state_dicts = torch.load(model_loc)
upuppi_state_dict = state_dicts['model']
upuppi.load_state_dict(upuppi_state_dict)    
print("Model loaded from {}".format(model_loc))


def embedding_loss(data, pfc_enc, vtx_enc):
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

        if i//10 == 0:
            random_indices = torch.randperm(len(truth_batch))[:30]
            random_vtx_encoding = vertex_encoding[random_indices, :]
            for j in range(len(random_vtx_encoding)):
                for k in range(j+1, len(random_vtx_encoding)):
                    vtx_loss = -0.01*euclidean_loss(random_vtx_encoding[j, :], random_vtx_encoding[k, :])
                    total_vtx_loss += vtx_loss
        else:
            continue
            
            # regularize the whole embedding to keep it normalized
    reg_loss = ((torch.norm(vtx_enc, p=2, dim=1)/10)**6).mean()
    # print the losses
    # print("Pfc loss: ", total_pfc_loss.item(), " Vtx loss: ", total_vtx_loss.item(), " Reg loss: ", reg_loss.item())
    return total_pfc_loss + total_vtx_loss + reg_loss


def process_data(data):
    '''
    Apply data processing as needed and return the processed data.
    '''
    return data
    data.x_pfc[:, -1] *= -1
    data.y *= -1
    neutral_indices = torch.nonzero(data.x_pfc[:,-2] == 0).squeeze()
    charged_indices = torch.nonzero(data.x_pfc[:,-2] != 0).squeeze()
    # convert z of neutral particles from -199 to 0
    # data.x_pfc[neutral_indices, -1] *= -1
    return data


def train(c_ratio=0.05, neutral_ratio=1):
    upuppi.train()
    counter = 0
    total_loss = 0
    for data in tqdm(train_loader):
        # euclidean_loss = nn.MSELoss().to(device)
        # let euclidean_loss be the L1 loss
        euclidean_loss = nn.L1Loss().to(device)
        counter += 1
        data = data.to(device) 
        data = process_data(data)
        optimizer.zero_grad()
        out, batch, pfc_enc, vtx_enc = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        if c_ratio > 0:
            emb_loss = (1/200)*embedding_loss(data, pfc_enc, vtx_enc)
        else:
            emb_loss = 0
        if neutral_ratio > 1:
            # calculate neutral loss
            neutral_indices = torch.nonzero(data.x_pfc[:, -2] == 0).squeeze()
            neutral_out = out[:,0][neutral_indices]
            neutral_y = data.y[neutral_indices]
            neutral_loss = euclidean_loss(neutral_out, neutral_y)
            # calculate charged loss
            charged_indices = torch.nonzero(data.x_pfc[:,-2] != 0).squeeze()
            charged_out = out[:,0][charged_indices]
            charged_y = data.y[charged_indices]
            charged_loss = euclidean_loss(charged_out, charged_y)
            # calculate total loss
            regression_loss = 200*(neutral_ratio*neutral_loss + charged_loss)/(neutral_ratio + 1)
        else:
            regression_loss = 200*euclidean_loss(out.squeeze(), data.y)
        if counter % 50 == 0:
            print("Regression loss: ", regression_loss.item(), " Embedding loss: ", emb_loss)
        loss = (c_ratio*emb_loss) + (1-c_ratio)*regression_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
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
        out, batch, pfc_enc, vtx_enc = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        regression_loss = euclidean_loss(out.squeeze(), data.y)
        loss = regression_loss
        total_loss += loss.item()
    total_loss = total_loss / counter        
    return total_loss

NUM_EPOCHS = 5

for epoch in range(epoch_to_load+1, NUM_EPOCHS+epoch_to_load+1): 
    loss = 0
    test_loss = 0
    if epoch % 2 == 1:
        c_ratio = 0.05
    else:
        c_ratio=0
    loss = train(c_ratio=c_ratio, neutral_ratio=epoch+1)
    state_dicts = {'model':upuppi.state_dict(),
                   'opt':optimizer.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
    print("Model saved")
    print("Time elapsed: ", time.time() - start_time)
    print("-----------------------------------------------------")
    # test_loss = test()
    print("Epoch: ", epoch, " Loss: ", loss)

    

