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
from torch import optim
from models.modelv2 import Net
from tqdm import tqdm
import copy


BATCHSIZE = 64
start_time = time.time()
data_train = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/train/")
data_test = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/test/")

# data_train = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/train2/")
# data_test = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/test2/")


train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_vtx'])

model = "combined_model"
model = "Dynamic_GATv2"
model = "modelv2"
# model = "modelv3"
model_dir = '/work/submit/cfalor/upuppi/deepjet-geometric/models/{}/'.format(model)
#model_dir = '/home/yfeng/UltimatePuppi/deepjet-geometric/models/v0/'


print("Training {}...".format(model))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# print the device used
print("Using device: ", device, torch.cuda.get_device_name(0))


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





def train(optimizer, upuppi, c_ratio=0.05, neutral_ratio=1):
    upuppi.train()
    counter = 0
    total_loss = 0
    for data in tqdm(train_loader):
        counter += 1
        data = data.to(device) 
        optimizer.zero_grad()
        out, batch, pfc_enc, vtx_enc = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        if c_ratio > 0:
            emb_loss = (1/200)*embedding_loss(data, pfc_enc, vtx_enc)
        else:
            emb_loss = 0
        if neutral_ratio > 1:
            # calculate neutral loss
            neutral_indices = torch.nonzero(data.x_pfc[:, 11] == 0).squeeze()
            neutral_out = out[:,0][neutral_indices]
            neutral_y = data.y[neutral_indices]
            neutral_loss = nn.MSELoss()(neutral_out, neutral_y)
            # calculate charged loss
            charged_indices = torch.nonzero(data.x_pfc[:,11] != 0).squeeze()
            charged_out = out[:,0][charged_indices]
            charged_y = data.y[charged_indices]
            charged_loss = nn.MSELoss()(charged_out, charged_y)
            # calculate total loss
            regression_loss = 200*(neutral_ratio*neutral_loss + charged_loss)/(neutral_ratio + 1)
        else:
            regression_loss = 200*nn.MSELoss()(out.squeeze(), data.y)
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
def test(upuppi):
    upuppi.eval()
    euclidean_loss = nn.MSELoss()
    counter = 0
    total_loss = 0
    for data in tqdm(test_loader):
        counter += 1
        data = data.to(device)
        out, batch, pfc_enc, vtx_enc = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        regression_loss = euclidean_loss(out.squeeze(), data.y)
        loss = regression_loss
        total_loss += loss.item()
    total_loss = total_loss / counter        
    return total_loss

def hyperparameter_search():
    # define the hyperparameter search space
    c_ratios = np.logspace(-3, -1, 3)
    neutral_ratios = np.linspace(1, 10, 3)
    lr = np.logspace(-4, -1, 3)
    hidden_dims = np.logspace(0.5, 2, 3).astype(int)
    k1s = np.logspace(0.6, 1.7, 3).astype(int)
    k2s = np.logspace(0.6, 1.7, 3).astype(int)
    dropouts = np.linspace(0, 0.5, 3)
    optimizers = ['adam', 'sgd', 'adagrad', 'adadelta', 'rmsprop']
    # define the search space
    search_space = {'c_ratio': c_ratios, 'neutral_ratio': neutral_ratios, 'lr': lr, 'hidden_dim': hidden_dims, 'dropout': dropouts, 'optimizer': optimizers}
    
    
    # define a function which uses binary search to find the best hyperparameters
    # train the model
    NUM_EPOCHS = 5
    # define the best model parameters
    best_hyperparameters = {'c_ratio': 0.05, 'neutral_ratio': 1, 'lr': 0.001, 'hidden_dim': 100, 'dropout': 0,'k1': 1, 'k2': 1, 'optimizer': 'adam','best_loss': 1000000, 'epoch': 0, 'best_model': None}
    hyperparameter_list = []

    for optimizer_type in optimizers:
        for c_ratio in c_ratios:
                for neutral_ratio in neutral_ratios:
                    for lr in lr:
                        for dropout in dropouts:
                            for hidden_dim in hidden_dims:
                                for k1 in k1s:
                                    for k2 in k2s:
                                        upuppi = Net(hidden_dim=hidden_dim, dropout=dropout).to(device)
                                        print("Training with: c_ratio: ", c_ratio, " neutral_ratio: ", neutral_ratio, " lr: ", lr, " hidden_dim: ", hidden_dim, " dropout: ", dropout, " k1: ", k1, " k2: ", k2, " optimizer: ", optimizer_type)
                                        # define the optimizer
                                        if optimizer_type == 'adam':
                                            optimizer = optim.Adam(upuppi.parameters(), lr=lr)
                                        elif optimizer_type == 'sgd':
                                            optimizer = optim.SGD(upuppi.parameters(), lr=lr)
                                        elif optimizer_type == 'adagrad':
                                            optimizer = optim.Adagrad(upuppi.parameters(), lr=lr)
                                        elif optimizer_type == 'adadelta':
                                            optimizer = optim.Adadelta(upuppi.parameters(), lr=lr)
                                        elif optimizer_type == 'rmsprop':
                                            optimizer = optim.RMSprop(upuppi.parameters(), lr=lr)

                                        # train the model
                                        for epoch in range(NUM_EPOCHS):
                                            print("Epoch: ", epoch)
                                            train(optimizer, upuppi, c_ratio=c_ratio, neutral_ratio=neutral_ratio)
                                            # test the model
                                            test_loss = test(upuppi)
                                            print("Test loss: ", test_loss)
                                            hyperparameter_dict = {'c_ratio': c_ratio, 'neutral_ratio': neutral_ratio, 'lr': lr, 'hidden_dim': hidden_dim, 'dropout': dropout, 'k1': k1, 'k2': k2, 'optimizer': optimizer_type, 'loss': test_loss, 'epoch': epoch}
                                            hyperparameter_list.append(hyperparameter_dict)
                                            # save the list
                                            with open('hyperparameter_list.txt', 'w') as f:
                                                f.write(str(hyperparameter_list))
                                            # check if the model has the best loss
                                            if test_loss < best_hyperparameters['best_loss']:
                                                # update the best hyperparameters dict
                                                best_hyperparameters['c_ratio'], best_hyperparameters['neutral_ratio'], best_hyperparameters['lr'], best_hyperparameters['hidden_dim'], best_hyperparameters['dropout'], best_hyperparameters['k1'], best_hyperparameters['k2'], best_hyperparameters['optimizer'], best_hyperparameters['epoch'], best_hyperparameters['best_model'] = c_ratio, neutral_ratio, lr, hidden_dim, dropout, k1, k2, optimizer_type, epoch, upuppi
                                                best_hyperparameters['best_loss'] = test_loss
                                                best_model = copy.deepcopy(upuppi)
                                                print("Best loss: ", best_hyperparameters['best_loss'], " Hyperparameters: ", best_hyperparameters)
                                                # save the best model
                                                torch.save(best_model.state_dict(), os.path.join(model_dir, "best_model.pt"))
                                                # save the best model parameters
                                                with open("best_model_parameters.txt", "w") as f:
                                                    f.write(str(best_hyperparameters))

if __name__ == "__main__":
    hyperparameter_search()


