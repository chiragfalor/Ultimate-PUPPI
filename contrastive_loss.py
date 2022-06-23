import time, os, torch, numpy as np
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
from torch import nn
from torch.nn import functional as F
# from models.embedding_model import Net
from models.embedding_GCN import Net
from tqdm import tqdm


BATCHSIZE = 1
start_time = time.time()
print("Training...")
data_train = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/train/")
data_test = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/test/")


train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])

model = "contrastive_loss"
model = "embedding_GCN"
model = "embedding_GCN_v1"
model = "embedding_GCN_cheating"
model = "embedding_GCN_cheating_low_lr"
model_dir = '/work/submit/cfalor/upuppi/deepjet-geometric/models/{}/'.format(model)
#model_dir = '/home/yfeng/UltimatePuppi/deepjet-geometric/models/v0/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# print the device used
print("Using device: ", device, torch.cuda.get_device_name(0))

# create the model
net = Net(pfc_input_dim=13).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


def contrastive_loss(pfc_enc, vtx_id, num_pfc=64, c=1.0, print_bool=False):
    '''
    Calculate the contrastive loss
    input:
    pfc_enc: the encodding of the inputs
    vtx_id: the true vertex which the particle is connected to
    num_pfc: number of particles to randomly sample
    c: the ratio of positive factor for the particles of same vertex divided by the negative factor
    '''
    # loss which encourages the embedding of same particles to be close and different particles to be far
    # randomly select a set of particles to be used for contrastive loss
    random_perm = torch.randperm(len(pfc_enc))
    if len(pfc_enc) < 2*num_pfc:
        num_pfc = len(pfc_enc)//2
        # print(len(pfc_enc))
    random_indices1 = random_perm[:num_pfc]
    random_indices2 = random_perm[num_pfc:2*num_pfc]
    pfc_enc_1 = pfc_enc[random_indices1, :]
    pfc_enc_2 = pfc_enc[random_indices2, :]
    vtx_id_1 = vtx_id[random_indices1]
    vtx_id_2 = vtx_id[random_indices2]
    # get a mask which is c if the particles are the same and -1 if they are different
    mask = -1+(c+1)*(vtx_id_1 == vtx_id_2).float()
    euclidean_dist = F.pairwise_distance(pfc_enc_1, pfc_enc_2)
    loss = torch.mean(mask*torch.pow(euclidean_dist, 2))
    if print_bool:
        print("Contrastive loss: {}, loss from particles: {}".format(loss, torch.mean(mask*torch.pow(euclidean_dist, 2))))
    return loss

def contrastive_loss_v2(pfc_enc, vtx_id, c=0.5, print_bool=False):
    unique_vtx = torch.unique(vtx_id)
    mean_vtx = torch.zeros((len(unique_vtx), pfc_enc.shape[1])).to(device)
    for i, vtx in enumerate(unique_vtx):
        mean_vtx[i] = torch.mean(pfc_enc[vtx_id == vtx, :], dim=0)
    # get the mean of the particles of the different vertex
    mean_vtx_diff = torch.zeros((len(unique_vtx), pfc_enc.shape[1])).to(device)
    for i, vtx in enumerate(unique_vtx):
        mean_vtx_diff[i] = torch.mean(pfc_enc[vtx_id != vtx, :], dim=0)
    # get the distance between the mean of the particles of the same vertex and the mean of the particles of the different vertex
    euclidean_dist_vtx = F.pairwise_distance(mean_vtx, mean_vtx_diff)
    loss = -torch.mean(torch.pow(euclidean_dist_vtx, 2))
    # add variance of the particles of the same vertex
    var_vtx = torch.zeros((len(unique_vtx), pfc_enc.shape[1])).to(device)
    for i, vtx in enumerate(unique_vtx):
        var_vtx[i] = torch.var(pfc_enc[vtx_id == vtx, :], dim=0)
    loss += c*torch.mean(torch.pow(var_vtx, 2))
    # print all the losses, the loss from the different means and the loss from the variance
    if print_bool:
        print("Contrastive loss: {}, loss from vtx distance: {}, loss from variance: {}".format(loss, -torch.mean(torch.pow(euclidean_dist_vtx, 2)), c*torch.mean(torch.pow(var_vtx, 2))))
    return loss
        


def process_batch(data):
    '''
    Process the batch of data
    input:
    data: the data batch
    output:
    data: the processed data batch
    '''
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
    loss_fn = contrastive_loss_v2
    train_loss = 0
    for counter, data in enumerate(tqdm(train_loader)):
        data = data.to(device)
        # data = process_batch(data)
        optimizer.zero_grad()
        vtx_id = (data.truth != 0).int()
        # adding in the true vertex id itself to check if model is working
        input_data = torch.cat((data.x_pfc[:,:-1], vtx_id.unsqueeze(1)), dim=1)
        # input_data = data.x_pfc
        pfc_enc = net(input_data)
        if neutral_weight != 1:
            charged_idx, neutral_idx = torch.nonzero(data.x_pfc[:,11] != 0).squeeze(), torch.nonzero(data.x_pfc[:,11] == 0).squeeze()
            charged_embeddings, neutral_embeddings = pfc_enc[charged_idx], pfc_enc[neutral_idx]
            charged_loss, neutral_loss = loss_fn(charged_embeddings, vtx_id[charged_idx], print_bool=False), loss_fn(neutral_embeddings, vtx_id[neutral_idx], print_bool=False)
            loss = (charged_loss + neutral_weight*neutral_loss)/(1+neutral_weight)
            loss += loss_fn(pfc_enc, vtx_id, print_bool=False)
        else:
            loss = loss_fn(pfc_enc, vtx_id, c=0.1, print_bool=False)
        loss += reg_ratio*((torch.norm(pfc_enc, p=2, dim=1))**4).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if counter % 5000 == 1:
            print("Counter: {}, Average Loss: {}".format(counter, train_loss/counter))
            print("Regression loss: {}".format(((torch.norm(pfc_enc, p=2, dim=1))**4).mean()))
            if neutral_weight != 1:
                print("Charged loss: {}, Neutral loss: {}".format(charged_loss, neutral_loss))
                print("number of charged particles: {}, number of neutral particles: {}".format(len(charged_idx), len(neutral_idx)))
            # loss = contrastive_loss(pfc_enc, vtx_id, num_pfc=64, c=0.1, print_bool=True)
            loss = loss_fn(pfc_enc, vtx_id, c=0.1, print_bool=True)
    train_loss = train_loss/counter
    return train_loss

# test function
@torch.no_grad()
def test():
    net.eval()
    test_loss = 0
    for counter, data in enumerate(train_loader):
        data = data.to(device)
        pfc_enc = net(data.x_pfc)
        vtx_id = data.truth
        loss = contrastive_loss(pfc_enc, vtx_id)
        test_loss += loss.item()
    test_loss = test_loss / counter
    return test_loss

# train the model

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

    

