from torch_cluster import knn
import torch
import numpy as np
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader

# random seed
np.random.seed(0)
torch.manual_seed(40)

# file for debugging tests

# get home directory path
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()

if __name__ == '__main__':
    # test visualize_embeddings
    # load the model
    data_test = UPuppiV0("/work/submit/cfalor/upuppi/Ultimate-PUPPI/test/")
    model = "embedding_model"
    model = "GravNetConv"
    # model = "combined_model"
    # model = "combined_model2"
    model = "modelv2"
    # model = "modelv2_neg"
    # model = "modelv2_nz199"
    # model = "modelv2_nz0"
    # model = "modelv2_orig"
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
    elif model == "modelv2" or model == "modelv2_neg" or model == "modelv2_nz0" or model == "modelv2_nz199" or model == "modelv2_orig":
        from models.modelv2 import Net
    elif model == "modelv3":
        from models.modelv3 import Net
    elif model == "Dynamic_GATv2":
        from models.Dynamic_GATv2 import Net
    else:
        raise(Exception("Model not found"))

    test_loader = DataLoader(data_test, batch_size=32, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
    model_dir = home_dir + 'models/{}/'.format(model)

    # load the model
    epoch_num = 20
    upuppi_state_dict = torch.load(model_dir + 'epoch-{}.pt'.format(epoch_num))['model']
    net = Net(pfc_input_dim=13, hidden_dim=256, k1 = 64, k2 = 12)
    net.load_state_dict(upuppi_state_dict)
    net.eval() 
    with torch.no_grad():
        data = next(iter(test_loader))
        out, batch, pfc_embeddings, vtx_embeddings = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
    print(pfc_embeddings.shape)
    print(batch)
    print(pfc_embeddings)
    edges = knn(pfc_embeddings, pfc_embeddings, 10, batch, batch)
    print(edges.shape)
    print(edges)