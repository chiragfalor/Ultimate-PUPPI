import sklearn
import numpy as np
import sys
#sys.path.append('/home/yfeng/UltimatePuppi/deepjet-geometric/')
from deepjet_geometric.datasets import UPuppiV0
from torch_geometric.data import DataLoader
import os

BATCHSIZE = 30

data_train = UPuppiV0("/work/submit/bmaier/upuppi/data/v0_z_regression/train/")
data_test = UPuppiV0("/work/submit/bmaier/upuppi/data/v0_z_regression/test/")
#data_train = UPuppiV0("/home/yfeng/UltimatePuppi/deepjet-geometric/data/train/")
#data_test = UPuppiV0("/home/yfeng/UltimatePuppi/deepjet-geometric/data/test/")

train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_vtx'])

import torch
from torch import nn
from models.model import Net

#import utils

model_dir = '/home/submit/ishenogi/upuppi/deepjet-geometric/examples/models/v0/'
#model_dir = '/home/yfeng/UltimatePuppi/deepjet-geometric/models/v0/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

upuppi = Net(isFCN=True).to(device)
#puma.load_state_dict(torch.load(model_dir+"epoch-32.pt")['model'])
optimizer = torch.optim.Adam(upuppi.parameters(), lr=0.001)
#optimizer.load_state_dict(torch.load(model_dir+"epoch-32.pt")['opt'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#scheduler.load_state_dict(torch.load(model_dir+"epoch-32.pt")['lr'])

def train():
    upuppi.train()
    counter = 0

    total_loss = 0
    for data in train_loader:
        counter += 1
        #print(str(counter*BATCHSIZE)+' / '+str(len(train_loader.dataset)))
        data = data.to(device)
        optimizer.zero_grad()
        out = upuppi(data.x_pfc,
                    data.x_vtx,
                    data.x_pfc_batch,
                    data.x_vtx_batch)
        
        #print('x_pfc')
        #print(data.x_pfc)
        #print(data.x_pfc[0].shape)
        #print(data.x_pfc[1].shape)
        #print('model:')
        #print(torch.squeeze(out[0]).view(-1))
        #print('target:')
        #print(data.y)

        #print("Predicted values on training dataset: ", torch.squeeze(out[0]).view(-1)[data.x_pfc[:,-1]==0])
        #print("Actual values in training dataset: ", data.y[data.x_pfc[:,-1]==0])	

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
        with torch.no_grad():
            out = upuppi(data.x_pfc,
                        data.x_vtx,
                        data.x_pfc_batch,
                        data.x_vtx_batch)
            #print("Predicted values on validation dataset: ", torch.squeeze(out[0]).view(-1)[data.x_pfc[:,-2]!=0])
            #print("In the dataset: ", data.x_pfc[data.x_pfc[:,-2]!=0,-1])
            #print("Actual values in validtion dataset: ",  data.y[data.x_pfc[:,-2]!=0])

            #loss = nn.MSELoss()(torch.squeeze(out[0]).view(-1)[data.x_pfc[:,-2]!=0], data.y[data.x_pfc[:,-2]!=0])
            loss = nn.MSELoss()(out[0][:,0], data.y)
            total_loss += loss.item()
    return total_loss / len(test_loader.dataset)


for epoch in range(1, 20):
    loss = train()
    scheduler.step()
    loss_val = test()

    print('Epoch {:03d}, Loss: {:.8f}, Val_loss: {:.8f}'.format(
        epoch, loss, loss_val))

    state_dicts = {'model':upuppi.state_dict(),
                   'opt':optimizer.state_dict(),
                   'lr':scheduler.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))






#Create csv file

import pandas as pd

final_ftp = None
final_ftt = None
final_f_pt = None
final_f_eta = None
final_f_phi = None
final_f_e = None
final_f_pid = None
final_f_charge = None
final_f_inputz = None

upuppi.eval()
total_loss = 0
counter = 0
for data in test_loader:
    counter += 1
    data = data.to(device)
    with torch.no_grad():
        out = upuppi(data.x_pfc,
                     data.x_vtx,
                     data.x_pfc_batch,
                     data.x_vtx_batch)
        
        if final_ftp is None:
            final_ftp = torch.squeeze(out[0]).view(-1).cpu().numpy()
            final_ftt = data.y.cpu().numpy()
            final_f_pt = data.x_pfc[:,0].cpu().numpy()
            final_f_eta = data.x_pfc[:,1].cpu().numpy()
            final_f_phi = data.x_pfc[:,2].cpu().numpy()
            final_f_e = data.x_pfc[:,3].cpu().numpy()
            final_f_pid = data.x_pfc[:,4].cpu().numpy()
            final_f_charge = data.x_pfc[:,-2].cpu().numpy()
            final_f_inputz = data.x_pfc[:,-1].cpu().numpy()
        else:
            tmp_ftp = torch.squeeze(out[0]).view(-1).cpu().numpy()
            tmp_ftt = data.y.cpu().numpy()
            tmp_f_pt = data.x_pfc[:,0].cpu().numpy()
            tmp_f_eta = data.x_pfc[:,1].cpu().numpy()
            tmp_f_phi = data.x_pfc[:,2].cpu().numpy()
            tmp_f_e = data.x_pfc[:,3].cpu().numpy()
            tmp_f_pid = data.x_pfc[:,4].cpu().numpy()
            tmp_f_charge = data.x_pfc[:,-2].cpu().numpy()
            tmp_f_inputz = data.x_pfc[:,-1].cpu().numpy()

            final_ftp = np.concatenate((final_ftp,tmp_ftp),axis=0)
            final_ftt = np.concatenate((final_ftt,tmp_ftt),axis=0)
            final_f_pt = np.concatenate((final_f_pt,tmp_f_pt),axis=0)
            final_f_eta = np.concatenate((final_f_eta,tmp_f_eta),axis=0)
            final_f_phi = np.concatenate((final_f_phi,tmp_f_phi),axis=0)
            final_f_e = np.concatenate((final_f_e,tmp_f_e),axis=0)
            final_f_pid = np.concatenate((final_f_pid,tmp_f_pid),axis=0)
            final_f_charge = np.concatenate((final_f_charge,tmp_f_charge),axis=0)
            final_f_inputz = np.concatenate((final_f_inputz,tmp_f_inputz),axis=0)

datadict = {'zpred':final_ftp,'ztrue':final_ftt,'input_pt':final_f_pt,'input_eta':final_f_eta,'input_phi':final_f_phi,'input_e':final_f_e,'input_pid':final_f_pid,
        'input_charge':final_f_charge,'input_z':final_f_inputz}
df = pd.DataFrame.from_dict(datadict)
df.to_csv("/work/submit/bmaier/upuppi/results/finalcsv.txt")
#df.to_csv("/home/yfeng/UltimatePuppi/deepjet-geometric/results/finalcsv.txt")

'''

import csv
ftp = open('train_z_preds.csv', 'w' )
writertp = csv.writer(ftp)
ftt = open('train_z_truth.csv', 'w')
writertt = csv.writer(ftt)

upuppi.train()
counter = 0

for data in train_loader:
    counter += 1
    data = data.to(device)
    optimizer.zero_grad()
    out = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
    writertp.writerow(torch.squeeze(out[0]).view(-1)[data.x_pfc[:,-1]==0])
    print('train_z_preds: ', torch.squeeze(out[0]).view(-1)[data.x_pfc[:,-1]==0])
    writertt.writerow(data.y[data.x_pfc[:,-1]==0])
ftp.close()
ftt.close()
'''
