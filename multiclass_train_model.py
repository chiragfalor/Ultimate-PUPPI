import time
from helper_functions import *
from loss_functions import *

start_time = time.time()

data_train = UPuppiV0(home_dir + 'train/')
data_test = UPuppiV0(home_dir + 'test/')
BATCHSIZE = 64

model_name = "multiclassifier_puppi_2_vtx_weighted"

vtx_classes = 2


print("Training {}...".format(model_name))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("Using device: ", device, torch.cuda.get_device_name(0))

model_dir = home_dir + 'models/{}/'.format(model_name)
net = get_neural_net(model_name)(dropout=0, vtx_classes=vtx_classes).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# save the model hyperparameters in the model directory
if not os.path.exists(model_dir): os.makedirs(model_dir)
with open(model_dir + 'hyperparameters.txt', 'w') as f: 
    f.write("network_architecture: {}\n".format(net))



train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])



def train(model, optimizer, loss_fn, embedding_loss_weight=0.1, neutral_weight = 1):
    '''
    Trains the given model for one epoch
    '''
    model.train()
    train_loss = 0
    for counter, data in enumerate(tqdm(train_loader)):
        data = process_data(data)
        data.to(device)
        optimizer.zero_grad()
        scores, pfc_embeddings, vtx_embeddings = model(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        # scores = scores.squeeze()
        vtx_embeddings = None  # uncomment if you want to use contrastive loss
        loss = loss_fn(data, scores, pfc_embeddings, vtx_embeddings, embedding_loss_weight, neutral_weight, vtx_classes=vtx_classes)
        # if loss is nan, print everything
        if np.isnan(loss.item()):
            print("Loss is nan")
            continue
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if counter % 100 == 1:
            print("Train loss: ", train_loss / counter)
            loss_fn(data, scores, pfc_embeddings, vtx_embeddings=vtx_embeddings, embedding_loss_weight=embedding_loss_weight, neutral_weight=neutral_weight, print_bool = True)
    return train_loss / counter

@torch.no_grad()
def test(model):
    '''
    Tests the given model on the test set
    '''
    model.eval()
    test_accuracy = 0
    test_neutral_accuracy = 0
    for counter, data in enumerate(tqdm(test_loader)):
        data = process_data(data)
        data.to(device)
        score = model(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)[0]
        score = score.squeeze()
        pred = (score[:, 0] < 0).int()
        accuracy = (pred == (data.truth != 0).int()).float().mean()
        neutral_mask = (data.x_pfc[:, -2] == 0)
        accuracy_neutral = (pred[neutral_mask] == (data.truth[neutral_mask] != 0).int()).float().mean()
        test_neutral_accuracy += accuracy_neutral.item()
        test_accuracy += accuracy.item()
    return test_accuracy / counter, test_neutral_accuracy / counter


NUM_EPOCHS = 40

model_performance = []

for epoch in range(NUM_EPOCHS):
    if epoch % 2 == 0:
        embedding_loss_weight = 0.02
    else:
        embedding_loss_weight = 0.0
    train_loss = train(net, optimizer, loss_fn=combined_classification_embedding_loss_puppi, embedding_loss_weight=embedding_loss_weight, neutral_weight=epoch+1)
    state_dicts = {'model':net.state_dict(),
                    'opt':optimizer.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{:02d}.pt'.format(epoch)))
    print("Model saved")
    print("Time elapsed: ", time.time() - start_time)
    print("-----------------------------------------------------")
    test_accuracy, test_neutral_accuracy = test(net)
    print("Epoch: {:02d}, Train Loss: {:4f}, Test Accuracy: {:.2%}, Test Neutral Accuracy: {:.2%}".format(epoch, train_loss, test_accuracy, test_neutral_accuracy))
    model_performance.append({'epoch':epoch, 'train_loss':train_loss, 'test_accuracy':test_accuracy, 'test_neutral_accuracy':test_neutral_accuracy})
# save the model performance as txt
with open(model_dir + 'model_performance.txt', 'w') as f:
    for item in model_performance:
        f.write("{}\n".format(item))

epoch_to_load = NUM_EPOCHS - 1
import load_pred_save
