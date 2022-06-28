# pyright: reportMissingModuleSource=false

import time
#sys.path.append('/home/yfeng/UltimatePuppi/Ultimate-PUPPI/')
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
import os
import torch
from torch import nn
from models.modelv2 import Net
from tqdm import tqdm
from contrastive_loss import contrastive_loss, contrastive_loss_v2


BATCHSIZE = 32
start_time = time.time()
# load home directory path from home_path.txt
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()

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
model = "modelv2_contrastive"
model_dir = home_dir + 'models/{}/'.format(model)
#model_dir = '/home/yfeng/UltimatePuppi/deepjet-geometric/models/v0/'


print("Training {}...".format(model))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# print the device used
print("Using device: ", device, torch.cuda.get_device_name(0))

# create the model
upuppi = Net(pfc_input_dim=13, hidden_dim=256, k1 = 64, k2 = 12).to(device)
optimizer = torch.optim.Adam(upuppi.parameters(), lr=0.001)


def process_data(data):
    '''
    Apply data processing as needed and return the processed data.
    '''
    return data
    # switch the sign of the z coordinate of the pfc
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
        euclidean_loss = nn.MSELoss().to(device)
        # let euclidean_loss be the L1 loss
        euclidean_loss = nn.L1Loss().to(device)
        counter += 1
        data = data.to(device)
        data = process_data(data)
        vtx_id = data.truth.int()
        optimizer.zero_grad()
        out, batch, pfc_enc, vtx_enc = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        if c_ratio > 0:
            emb_loss = contrastive_loss_v2(pfc_enc, vtx_id, c=10)
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
            regression_loss = (neutral_ratio*neutral_loss + charged_loss)/(neutral_ratio + 1)
        else:
            regression_loss = euclidean_loss(out.squeeze(), data.y)
        # print(counter)
        if counter % 50 == 0:
            print("Regression loss: ", regression_loss.item(), " Embedding loss: ", emb_loss)
            contrastive_loss_v2(pfc_enc, vtx_id, print_bool=True)
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

# train the model
NUM_EPOCHS = 20

for epoch in range(1, NUM_EPOCHS+1): 
    loss = 0
    test_loss = 0
    if epoch % 2 == 1:
        c_ratio = 0.05
    else:
        c_ratio=0
    loss = train(c_ratio=c_ratio, neutral_ratio=2*epoch-1)
    state_dicts = {'model':upuppi.state_dict(),
                   'opt':optimizer.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
    print("Model saved")
    print("Time elapsed: ", time.time() - start_time)
    print("-----------------------------------------------------")
    # test_loss = test()
    print("Epoch: ", epoch, " Loss: ", loss)

    

