import time, os
from helper_functions import *
from loss_functions import *

start_time = time.time()

data_train = UPuppiV0(home_dir + 'train/')
data_test = UPuppiV0(home_dir + 'test/')
BATCHSIZE = 64


model_name = "vtx_pred_model2"


print("Training {}...".format(model_name))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("Using device: ", device, torch.cuda.get_device_name(0))

model_dir = home_dir + 'models/{}/'.format(model_name)
model = get_neural_net(model_name)(dropout=0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# save the model hyperparameters in the model directory
if not os.path.exists(model_dir): os.makedirs(model_dir)
with open(model_dir + 'hyperparameters.txt', 'w') as f: 
    f.write("network_architecture: {}\n".format(model))



train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_vtx'])


def train(model, optimizer, loss_fn, neutral_weight = 1):
    '''
    Trains the given model for one epoch
    '''
    model.train()
    train_loss = 0
    for counter, data in enumerate(tqdm(train_loader)):
        data = process_data(data)
        data.to(device)
        optimizer.zero_grad()
        pfc_embeddings, vtx_embeddings = model(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        # vtx_embeddings = None  # uncomment if you want to use contrastive loss
        # loss = loss_fn(data, z_pred, pfc_embeddings, vtx_embeddings=vtx_embeddings, embedding_loss_weight=embedding_loss_weight, neutral_weight=neutral_weight)
        if vtx_embeddings is not None:
            loss = loss_fn(pfc_enc = pfc_embeddings, vtx_enc = vtx_embeddings, true_vtx_id = data.truth, pfc_batch = data.x_pfc_batch, vtx_batch = data.x_vtx_batch)
        else:
            loss = loss_fn(pfc_enc = pfc_embeddings, true_vtx_id = data.truth)
        # if loss is nan, print everything
        if np.isnan(loss.item()):
            print("Loss is nan")
            continue
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if counter % 100 == 1:
            print("Train loss: ", train_loss / counter)
            # loss_fn(data, z_pred, pfc_embeddings, vtx_embeddings=vtx_embeddings, embedding_loss_weight=embedding_loss_weight, neutral_weight=neutral_weight, print_bool = True)
    return train_loss / counter

@torch.no_grad()
def test(model, loss_fn):
    '''
    Tests the given model on the test set
    '''
    model.eval()
    test_loss = 0
    for counter, data in enumerate(tqdm(test_loader)):
        data = process_data(data)
        data.to(device)
        pfc_embeddings, vtx_embeddings = model(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        if vtx_embeddings is not None:
            loss = loss_fn(pfc_enc = pfc_embeddings, vtx_enc = vtx_embeddings, true_vtx_id = data.truth, pfc_batch = data.x_pfc_batch, vtx_batch = data.x_vtx_batch)
        else:
            loss = loss_fn(pfc_enc = pfc_embeddings, true_vtx_id = data.truth)
        test_loss += loss.item()
    return test_loss / counter


NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    if epoch % 2 == 0:
        embedding_loss_weight = 0.01
    else:
        embedding_loss_weight = 0.0
    train_loss = train(model, optimizer, loss_fn=embedding_loss)
    state_dicts = {'model':model.state_dict(),
                    'opt':optimizer.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{:02d}.pt'.format(epoch)))
    print("Model saved")
    print("Time elapsed: ", time.time() - start_time)
    print("-----------------------------------------------------")
    # test_loss = test(model, loss_fn=combined_loss_fn)
    print("Epoch: ", epoch, "Train loss: ", train_loss)#, "Test loss: ", test_loss)
