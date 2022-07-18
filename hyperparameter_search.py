from helper_functions import *
from loss_functions import *
import random, time, copy
from torch import optim

# set random seeds
random.seed(2)
np.random.seed(2)
torch.manual_seed(2)

start_time = time.time()

data_train = UPuppiV0(home_dir + 'train/')
data_test = UPuppiV0(home_dir + 'test/')
BATCHSIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("Using device: ", device, torch.cuda.get_device_name(0))



train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_vtx'])


def train(model, optimizer, loss_fn, embedding_loss_weight=0.1, neutral_weight = 1):
    '''
    Trains the given model for one epoch
    '''
    model.train()
    train_loss = 0
    for counter, data in enumerate(tqdm(train_loader)):
        data.to(device)
        optimizer.zero_grad()
        z_pred, batch, pfc_embeddings, vtx_embeddings = model(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        # vtx_embeddings = None  # uncomment if you want to use contrastive loss
        loss = loss_fn(data, z_pred, pfc_embeddings, vtx_embeddings=vtx_embeddings, embedding_loss_weight=embedding_loss_weight, neutral_weight=neutral_weight)
        # if loss is nan, print everything
        if np.isnan(loss.item()):
            print("Loss is nan")
            print("data: ", data)
            print("z_pred: ", z_pred)
            print("pfc_embeddings: ", pfc_embeddings)
            print("vtx_embeddings: ", vtx_embeddings)
            print("data.x_pfc: ", data.x_pfc)
            print("data.x_vtx: ", data.x_vtx)
            print("data.x_pfc_batch: ", data.x_pfc_batch)
            print("data.x_vtx_batch: ", data.x_vtx_batch)
            print("data.truth: ", data.truth)
            loss_fn(data, z_pred, pfc_embeddings, vtx_embeddings=vtx_embeddings, embedding_loss_weight=embedding_loss_weight, neutral_weight=neutral_weight, print_bool = True)
            # sys.exit()
            continue
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / counter

@torch.no_grad()
def test(model, loss='euclidean'):
    '''
    Tests the given model on the test set and returns the total loss, loss across neutrals, and loss on neutrals in pileup
    the loss fn should be a regression loss function like nn.MSELoss() or nn.L1Loss()
    '''
    model.eval()
    total_loss = 0
    neutral_loss = 0
    neutral_pileup_loss = 0
    for counter, data in enumerate(tqdm(test_loader)):
        data.to(device)
        z_pred, batch, pfc_embeddings, vtx_embeddings = model(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        # calculate euclidean loss
        if loss == 'euclidean':
            dist = (z_pred - data.y).pow(2)
        else:
            dist = (z_pred - data.y).abs()

        total_loss += dist.mean().item()
        neutral_idx = (data.x_pfc[:,-2] == 0)
        neutral_loss += dist[neutral_idx].mean().item()

        neutral_pileup_idx = (data.x_pfc[:,-2] == 0) & (data.truth != 0)
        neutral_pileup_loss += dist[neutral_pileup_idx].mean().item()

    return total_loss / counter, neutral_loss / counter, neutral_pileup_loss / counter



def hyperparameter_search():
    # define the hyperparameter search space
    embedding_loss_weights = np.logspace(-3, 0, 5)
    neutral_weights = np.logspace(0, 2, 5).astype(int)
    lrs = np.logspace(-5, -3, 5)
    hidden_dims = np.logspace(1.9, 2.8, 5).astype(int)
    k1s = np.logspace(1, 2, 5).astype(int)
    k2s = np.logspace(0.5, 1.8, 5).astype(int)
    dropouts = np.linspace(0, 0.5, 5)
    model_names = ['modelv2', 'DynamicPointTransformer']
    optimizers = ['adam', 'adagrad', 'adadelta']    # sgd gives nan loss, rmsprop is relatively bad
    search_space = {'embedding_loss_weight': embedding_loss_weights, 'neutral_weight': neutral_weights, 'lr': lrs, 'hidden_dim': hidden_dims, 'k1': k1s, 'k2': k2s, 'dropout': dropouts, 'model_name': model_names, 'optimizer': optimizers}

    NUM_EPOCHS = 10
    # initialize best loss
    best_hyperparameters = {'best_loss': 100000}
    hyperparameter_list = []

    for _ in range(200):
        # randomly sample a hyperparameter configuration
        hyperparameter_config = {key: random.choice(search_space[key]) for key in search_space}
        print("Hyperparameter config: ", hyperparameter_config)
        upuppi = get_neural_net(hyperparameter_config['model_name'], new_net=True)(pfc_input_dim = 12, hidden_dim=hyperparameter_config['hidden_dim'].item(), k1=hyperparameter_config['k1'].item(), k2=hyperparameter_config['k2'].item(), dropout=hyperparameter_config['dropout'].item()).to(device)
        print("Training with hyperparameters: ", hyperparameter_config)
        # define the optimizer
        if hyperparameter_config['optimizer'] == 'adam':
            optimizer = optim.Adam(upuppi.parameters(), lr=hyperparameter_config['lr'].item())
        elif hyperparameter_config['optimizer'] == 'adagrad':
            optimizer = optim.Adagrad(upuppi.parameters(), lr=hyperparameter_config['lr'].item())
        elif hyperparameter_config['optimizer'] == 'adadelta':
            optimizer = optim.Adadelta(upuppi.parameters(), lr=hyperparameter_config['lr'].item())

        # train the model
        for epoch in range(NUM_EPOCHS):
            print("Epoch: ", epoch)
            train_loss = train(upuppi, optimizer, loss_fn=combined_loss_fn, embedding_loss_weight=hyperparameter_config['embedding_loss_weight'], neutral_weight=hyperparameter_config['neutral_weight'])
            total_loss, neutral_loss, neutral_pileup_loss = test(upuppi, loss='euclidean')
            print("Test losses: ", total_loss, neutral_loss, neutral_pileup_loss)
            # add in the losses and epoch to hyperparameter_dict
            hyperparameter_config['train_loss'], hyperparameter_config['total_loss'], hyperparameter_config['neutral_loss'], hyperparameter_config['neutral_pileup_loss'] = train_loss, total_loss, neutral_loss, neutral_pileup_loss
            hyperparameter_config['epoch'] = epoch
            hyperparameter_list.append(hyperparameter_config)
            # save the list
            with open('hyperparameter_list.txt', 'w') as f:
                f.write(str(hyperparameter_list))
            # check if the model has the best loss
            if total_loss < best_hyperparameters['best_loss']:
                # update the best hyperparameters dict
                for key in hyperparameter_config:
                    best_hyperparameters[key] = hyperparameter_config[key]
                best_model = copy.deepcopy(upuppi)
                print("Best loss: ", best_hyperparameters['best_loss'], "with hyperparameters: ", best_hyperparameters)
                # save the best model
                torch.save(best_model.state_dict(), os.path.join(home_dir, "best_model.pt"))
                # save the best model parameters
                with open("best_model_parameters.txt", "w") as f:
                    f.write(str(best_hyperparameters))

if __name__ == "__main__":
    hyperparameter_search()


