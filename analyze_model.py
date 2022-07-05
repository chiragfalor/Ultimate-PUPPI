from helper_functions import *

BATCHSIZE = 32

with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()
data_test = UPuppiV0(home_dir + 'train/')
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                             follow_batch=['x_pfc', 'x_vtx'])

# model_name = "DynamicGCN"
# # model_name = "GAT"
# # model_name = "GravNetConv"
# # model_name = "No_Encode_grav_net"
# model_name = "modelv2"
# model_name = "embedding_GCN_nocheating"
model_name = "modelv2_analysis"



upuppi = get_neural_net(model_name)(pfc_input_dim=13, hidden_dim=320, k1=32, k2=16, dropout=0)
# gets the model from models.modelv2
# print("Model architecture: ", upuppi)


model_dir = home_dir + 'models/{}/'.format(model_name)
epoch_to_load = 19
model_loc = os.path.join(model_dir, 'epoch-{:02d}.pt'.format(epoch_to_load))

# load model
state_dicts = torch.load(model_loc)
upuppi_state_dict = state_dicts['model']
optim_state_dict = state_dicts['opt']
print(upuppi_state_dict)
# print(optim_state_dict)


upuppi.load_state_dict(upuppi_state_dict)
upuppi.eval()

data = next(iter(test_loader))

z_pred, pfc_batch, pfc_embeddings, vtx_embeddings = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
# print(z_pred.shape)