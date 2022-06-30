import time
import numpy as np
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
import os
import torch
from tqdm import tqdm
from helper_functions import get_neural_net

BATCHSIZE = 32

    # data_test = UPuppiV0("/work/submit/cfalor/upuppi/z_reg/test/")
    # test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
    #                          follow_batch=['x_pfc', 'x_vtx'])

with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()

model = "DynamicGCN"
# model = "GAT"
# model = "GravNetConv"
# model = "No_Encode_grav_net"
model = "modelv2"
model = "embedding_GCN_nocheating"



upuppi = get_neural_net(model)()

#import utils

model_dir = home_dir + 'models/{}/'.format(model)
epoch_to_load = 0
model_loc = os.path.join(model_dir, 'epoch-{}.pt'.format(epoch_to_load))

# load model
state_dicts = torch.load(model_loc)
upuppi_state_dict = state_dicts['model']
upuppi.load_state_dict(upuppi_state_dict)
optim_state_dict = state_dicts['opt']
print(upuppi_state_dict)
# print(optim_state_dict)