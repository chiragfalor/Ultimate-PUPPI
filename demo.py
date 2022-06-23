import time
import sklearn
import numpy as np
import sys

import torch
from torch import nn
from models.model import Net
#sys.path.append('/home/yfeng/UltimatePuppi/deepjet-geometric/')
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
import os
from tqdm import tqdm

BATCHSIZE = 64
start_time = time.time()
data_train = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/train/")
data_test = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/test2/")
# data_train = UPuppiV0("/work/submit/bmaier/upuppi/data/v0_z_regression/train/")
# data_test = UPuppiV0("/work/submit/bmaier/upuppi/data/v0_z_regression/test/")
#data_train = UPuppiV0("/home/yfeng/UltimatePuppi/deepjet-geometric/data/train/")
#data_test = UPuppiV0("/home/yfeng/UltimatePuppi/deepjet-geometric/data/test/")

# train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
#                           follow_batch=['x_pfc', 'x_vtx'])
# load the dataset
# data_train = torch.load('/work/submit/cfalor/upuppi/z_reg/train/data_train.pt')
# data_test = torch.load('/work/submit/cfalor/upuppi/z_reg/test/data_test.pt')

print(data_train)
print(data_test)

# print type of data_train
print(type(data_train))


data_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])

print("Training...")

# get shape of dataset

# save the dataset
# torch.save(data_train, '/work/submit/cfalor/upuppi/z_reg/train/data_train.pt')
# torch.save(data_test, '/work/submit/cfalor/upuppi/z_reg/test/data_test.pt')

# print 5 random samples from the dataset

for batch_idx, data in enumerate(tqdm(data_loader)):
    print(data)
    print(data.x_pfc.shape)
    print(data.x_vtx)
    print(data.x_pfc_batch)
    print(data.y)
    print(batch_idx)
    if batch_idx == 5:
        break