import time
import numpy as np
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
import os
import torch
from tqdm import tqdm

BATCHSIZE = 32

    # data_test = UPuppiV0("/work/submit/cfalor/upuppi/z_reg/test/")
    # test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
    #                          follow_batch=['x_pfc', 'x_vtx'])


model = "DynamicGCN"
# model = "GAT"
# model = "GravNetConv"
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

upuppi = Net()

#import utils

model_dir = '/work/submit/cfalor/upuppi/z_reg/models/{}/'.format(model)
epoch_to_load = 1
model_loc = os.path.join(model_dir, 'epoch-{}.pt'.format(epoch_to_load))

# load model
state_dicts = torch.load(model_loc)
upuppi_state_dict = state_dicts['model']
upuppi.load_state_dict(upuppi_state_dict)

print(upuppi_state_dict)