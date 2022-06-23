# read model data and save predictions

import time
import numpy as np
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
import os
from torch import nn
import torch
from tqdm import tqdm




#import utils
def save_predictions(model, data_loader, model_name):
    """
    Save predictions to file.
    """
    upuppi = model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Using device: ", device, torch.cuda.get_device_name(0))
    upuppi.to(device)
    upuppi.eval()

    #Create csv file

    import pandas as pd

    # save final predictions and labels

    zpred = None
    ztrue = None
    px = None
    py = None
    eta = None
    E = None
    pid = None
    charge = None
    zinput = None

    upuppi.eval()
    total_loss = 0
    counter = 0
    for data in tqdm(test_loader):
        counter += 1
        data.to(device)
        with torch.no_grad():
            out = upuppi(data.x_pfc,data.x_vtx,data.x_pfc_batch,data.x_vtx_batch)
            loss = nn.MSELoss()(out[0][:,0], data.y)
            total_loss += loss.item()

            if zpred is None:
                zpred = torch.squeeze(out[0]).view(-1).cpu().numpy()
                ztrue = data.y.cpu().numpy()
                px = data.x_pfc[:,0].cpu().numpy()
                py = data.x_pfc[:,1].cpu().numpy()
                eta = data.x_pfc[:,2].cpu().numpy()
                E = data.x_pfc[:,3].cpu().numpy()
                # convert pid from one hot to int
                pid = np.argmax(data.x_pfc[:,4:11].cpu().numpy(), axis=1)
                charge = data.x_pfc[:,11].cpu().numpy()
                # zinput = data.x_pfc[:,12].cpu().numpy()
            else:
                zpred = np.concatenate((zpred, out[0][:,0].cpu().numpy()), axis=0)
                ztrue = np.concatenate((ztrue, data.y.cpu().numpy()), axis=0)
                px = np.concatenate((px, data.x_pfc[:,0].cpu().numpy()), axis=0)
                py = np.concatenate((py, data.x_pfc[:,1].cpu().numpy()), axis=0)
                eta = np.concatenate((eta, data.x_pfc[:,2].cpu().numpy()), axis=0)
                E = np.concatenate((E, data.x_pfc[:,3].cpu().numpy()), axis=0)
                # convert pid from one hot to int
                pid = np.concatenate((pid, np.argmax(data.x_pfc[:,4:11].cpu().numpy(), axis=1)), axis=0)
                charge = np.concatenate((charge, data.x_pfc[:,11].cpu().numpy()), axis=0)
                # zinput = np.concatenate((zinput, data.x_pfc[:,12].cpu().numpy()), axis=0)

        
    print("Total testing loss: {}".format(total_loss/(counter*BATCHSIZE)))  

    datadict = {'zpred':zpred, 'ztrue':ztrue, 'px':px, 'py':py, 'eta':eta, 'E':E, 'pid':pid, 'charge':charge, 'zinput':zinput}
    df = pd.DataFrame.from_dict(datadict)
    df.to_csv("/work/submit/cfalor/upuppi/deepjet-geometric/results/{}.csv".format(model_name), index=False)
    print("Saved predictions to file {}".format("/work/submit/cfalor/upuppi/deepjet-geometric/results/{}.csv".format(model_name)))


if __name__ == "__main__":
    BATCHSIZE = 64
    data_test = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/test/")
    test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                            follow_batch=['x_pfc', 'x_vtx'])


    epoch_to_load = 18
    # model = "DynamicGCN"
    # model = "GAT"
    model = "GravNetConv"
    # model = "No_Encode_grav_net"
    model = "combined_model"
    # model = "combined_model2"
    model = "modelv2"
    # model = "modelv3"
    # model = "Dynamic_GATv2"
    if model == "DynamicGCN":
        from models.DynamicGCN import Net
    elif model == "GAT":
        from models.GAT import Net
    elif model == "GravNetConv":
        from models.GravNetConv import Net
    elif model == "No_Encode_grav_net":
        from models.No_Encode_grav_net import Net
    elif model == "combined_model2" or model == "combined_model":
        from models.model import Net
    elif model == "modelv2":
        from models.modelv2 import Net
    elif model == "modelv3":
        from models.modelv3 import Net
    elif model == "Dynamic_GATv2":
        from models.Dynamic_GATv2 import Net
    else:
        raise(Exception("Model not found"))

    upuppi = Net(pfc_input_dim=13)
    model_dir = '/work/submit/cfalor/upuppi/deepjet-geometric/models/{}/'.format(model)
    model_loc = os.path.join(model_dir, 'epoch-{}.pt'.format(epoch_to_load))
    print("Saving predictions of model {}".format(model), "at epoch {}".format(epoch_to_load))

    # load model
    state_dicts = torch.load(model_loc)
    upuppi_state_dict = state_dicts['model']
    upuppi.load_state_dict(upuppi_state_dict)
    save_predictions(model=upuppi, data_loader=test_loader, model_name=model)   # save predictions