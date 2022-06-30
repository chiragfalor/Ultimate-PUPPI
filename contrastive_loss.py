# pyright: reportMissingImports=false
import re
import time, os, torch, numpy as np
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
from torch import nn
from torch.nn import functional as F
from helper_functions import get_neural_net, home_dir
from loss_functions import *
from tqdm import tqdm


BATCHSIZE = 1
start_time = time.time()
print("Training...")
data_train = UPuppiV0(home_dir + 'train/')
data_test = UPuppiV0(home_dir + 'test/')


train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])

model = "contrastive_loss"
model = "embedding_GCN"
model = "embedding_GCN_v1"
model = "embedding_GCN_cheating"
model = "embedding_GCN_cheating_low_lr"
model = "embedding_GCN_nocheating"
model = "embedding_GCN_allvtx"
model_dir = home_dir + 'models/{}/'.format(model)
#model_dir = '/home/yfeng/UltimatePuppi/deepjet-geometric/models/v0/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# print the device used
print("Using device: ", device, torch.cuda.get_device_name(0))

# create the model
net = get_neural_net(model)(pfc_input_dim=13).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)





def process_batch(data):
    '''
    Process the batch of data
    input:
    data: the data batch
    output:
    data: the processed data batch
    '''
    return data
    # get the data
    x_pfc = data.x_pfc.to(device)
    # normalize z to [-1, 1]
    data.x_pfc[:,12] = data.x_pfc[:,12]/200.0
    # zero out z
    data.x_pfc[:,12] = data.x_pfc[:,12]*0.0
    # normalize the true z to [-1, 1]
    data.y = data.y/200.0
    # return the data
    return data



def train(reg_ratio = 0.01, neutral_weight = 1):
    net.train()
    loss_fn = contrastive_loss
    train_loss = 0
    for counter, data in enumerate(tqdm(train_loader)):
        data = data.to(device)
        # data = process_batch(data)
        optimizer.zero_grad()
        # vtx_id = (data.truth != 0).int()
        vtx_id = data.truth.int()
        # adding in the true vertex id itself to check if model is working
        input_data = torch.cat((data.x_pfc[:,:-1], vtx_id.unsqueeze(1)), dim=1)
        charged_idx, neutral_idx = torch.nonzero(data.x_pfc[:,11] != 0).squeeze(), torch.nonzero(data.x_pfc[:,11] == 0).squeeze()
        # replace the vertex id of the neutral particles with 0.5
        # vtx_id[neutral_idx] = 0.5
        # input_data[neutral_idx, -1] = 0.5
        # input_data = data.x_pfc
        pfc_enc = net(input_data)
        # print(net.state_dict())
        # if pfc enc is nan, print the data
        if neutral_weight != 1:  
            charged_embeddings, neutral_embeddings = pfc_enc[charged_idx], pfc_enc[neutral_idx]
            charged_loss, neutral_loss = loss_fn(charged_embeddings, vtx_id[charged_idx], print_bool=False), loss_fn(neutral_embeddings, vtx_id[neutral_idx], print_bool=False)
            loss = (charged_loss + neutral_weight*neutral_loss)/(1+neutral_weight)
            loss += loss_fn(pfc_enc, vtx_id, c1=0.1, print_bool=False, reg_ratio=reg_ratio)
        else:
            loss = loss_fn(pfc_enc, vtx_id, c1=0.1, print_bool=False, reg_ratio=reg_ratio)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if counter % 5000 == 1:
            loss = loss_fn(pfc_enc, vtx_id, c1=0.1, print_bool=True, reg_ratio=reg_ratio)
            print("Counter: {}, Average Loss: {}".format(counter, train_loss/counter))
            if neutral_weight != 1:
                print("Charged loss: {}, Neutral loss: {}".format(charged_loss, neutral_loss))
                print("number of charged particles: {}, number of neutral particles: {}".format(len(charged_idx), len(neutral_idx)))
            # loss = contrastive_loss(pfc_enc, vtx_id, num_pfc=64, c=0.1, print_bool=True)
            
    train_loss = train_loss/counter
    return train_loss

# test function
@torch.no_grad()
def test():
    net.eval()
    test_loss = 0
    for counter, data in enumerate(tqdm(train_loader)):
        data = data.to(device)
        pfc_enc = net(data.x_pfc)
        vtx_id = data.truth
        loss = contrastive_loss_v2(pfc_enc, vtx_id)
        test_loss += loss.item()
    test_loss = test_loss / counter
    return test_loss

# train the model
if __name__ == "__main__":
    for epoch in range(20):
        loss = 0
        test_loss = 0
        loss = train(reg_ratio = 0.01, neutral_weight = epoch+1)
        state_dicts = {'model':net.state_dict(),
                    'opt':optimizer.state_dict()} 
        torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
        print("Model saved at path: {}".format(os.path.join(model_dir, 'epoch-{}.pt'.format(epoch))))
        print("Time elapsed: ", time.time() - start_time)
        print("-----------------------------------------------------")
        test_loss = test()
        print("Epoch: ", epoch, " Loss: ", loss, " Test Loss: ", test_loss)
