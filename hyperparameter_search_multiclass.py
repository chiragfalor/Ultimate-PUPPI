from helper_functions import *
from loss_functions import *
# from multiclass_train_model import train, test
import random, time, copy
from torch import optim

# set random seeds
random.seed(21)
np.random.seed(21)
torch.manual_seed(21)


start_time = time.time()

data_train = UPuppiV0(home_dir + 'train5/')
data_test = UPuppiV0(home_dir + 'test5/')
BATCHSIZE = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("Using device: ", device, torch.cuda.get_device_name(0))




def train(model, train_loader, optimizer, loss_fn, embedding_loss_weight=0.1, neutral_weight = 1, contrastive = True, classification_weighting = 'pt'):
    '''
    Trains the given model for one epoch
    '''
    model.train()
    train_loss = 0
    vtx_classes = None
    for counter, data in enumerate(tqdm(train_loader)):
        data = process_data(data)
        data.to(device)
        optimizer.zero_grad()
        scores, pfc_embeddings, vtx_embeddings = model(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        if vtx_classes is None:
            vtx_classes = scores.shape[1] - 1
        # scores = scores.squeeze()
        if contrastive:
            vtx_embeddings = None  # uncomment if you want to use contrastive loss
        loss = loss_fn(data, scores, pfc_embeddings, vtx_embeddings, embedding_loss_weight, neutral_weight, vtx_classes=vtx_classes, classification_weighting=classification_weighting)
        # if loss is nan, print everything
        if np.isnan(loss.item()):
            print("Loss is nan")
            continue
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if counter % 300 == 1:
            print("Train loss: ", train_loss / counter)
            loss_fn(data, scores, pfc_embeddings, vtx_embeddings=vtx_embeddings, embedding_loss_weight=embedding_loss_weight, neutral_weight=neutral_weight, print_bool = True)
    return train_loss / counter

@torch.no_grad()
def test(model, test_loader, loss_fn):
    model.eval()
    test_accuracy, test_neutral_accuracy, test_loss = 0, 0, 0
    all_pred_prob = None
    neutral_pred_prob = None
    for counter, data in enumerate(tqdm(test_loader)):
        data = process_data(data)
        data.to(device)
        score = model(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)[0]
        vtx_classes = score.shape[1] - 1
        loss = loss_fn(data, score, embedding_loss_weight=0, vtx_classes=vtx_classes, classification_weighting = 'pt')
        pred = torch.argmax(score, dim=1).long()
        pred_prob = torch.softmax(score, dim=1)
        truth = process_truth(data.truth, vtx_classes).long()
        neutral_mask = (data.x_pfc[:, -2] == 0)
        if all_pred_prob is None:
            all_pred_prob = pred_prob
            all_truth = truth
            neutral_pred_prob = pred_prob[neutral_mask]
            neutral_truth = truth[neutral_mask]
        else:
            all_pred_prob = torch.cat((all_pred_prob, pred_prob), dim=0)
            all_truth = torch.cat((all_truth, truth), dim=0)
            neutral_pred_prob = torch.cat((neutral_pred_prob, pred_prob[neutral_mask]), dim=0)
            neutral_truth = torch.cat((neutral_truth, truth[neutral_mask]), dim=0)
        accuracy = (pred == truth).float().mean()
        accuracy_neutral = (pred[neutral_mask] == truth[neutral_mask]).float().mean()
        test_loss += loss.item()
        test_neutral_accuracy += accuracy_neutral.item()
        test_accuracy += accuracy.item()
    fpr, tpr, thresholds = metrics.roc_curve(neutral_truth.cpu().numpy(), 1 - neutral_pred_prob.cpu().numpy()[:, 0])
    roc_auc = metrics.auc(fpr, tpr)
    return test_loss / counter, test_accuracy / counter, test_neutral_accuracy / counter, roc_auc


def hyperparameter_search():
    # define the hyperparameter search space
    embedding_loss_weights = np.logspace(-3, 0, 5)
    neutral_weights = np.logspace(0, 2, 5).astype(int)
    # lrs = np.logspace(-5, -3, 5)
    lrs = np.array([0.001])
    hidden_dims = np.logspace(1.5, 2.5, 5).astype(int)
    k1s = np.logspace(1, 2, 5).astype(int)
    k2s = np.logspace(0.5, 1.8, 5).astype(int)
    dropouts = np.linspace(0, 0.5, 5)
    cross_entropy_weighting = [None, 'pt', 'num']
    model_names = ['multiclass', 'deep_multiclass']
    # optimizers = ['adam', 'adagrad', 'adadelta']    # sgd gives nan loss, rmsprop is relatively bad
    optimizers = ['adam']
    contrastive_loss_fn = [True, False]  # contrastive loss or not
    search_space = {'embedding_loss_weight': embedding_loss_weights, 'neutral_weight': neutral_weights, 'lr': lrs, 'hidden_dim': hidden_dims, 'k1': k1s, 'k2': k2s, 'dropout': dropouts, 'model_name': model_names, 'optimizer': optimizers, 'cross_entropy_weighting': cross_entropy_weighting, 'contrastive': contrastive_loss_fn}

    NUM_EPOCHS = 10
    # initialize best loss
    best_hyperparameters = {'auc': 0}
    hyperparameter_list = []

    for _ in range(1000):
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
        if hyperparameter_config['hidden_dim'] >= 300:
            BATCHSIZE = 8
        else:
            BATCHSIZE = 16
        train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
        test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])

        # train the model
        for epoch in range(NUM_EPOCHS):
            print("Epoch: ", epoch)
            train_loss = train(upuppi, train_loader, optimizer, loss_fn=combined_classification_embedding_loss_puppi, embedding_loss_weight=hyperparameter_config['embedding_loss_weight'], neutral_weight=hyperparameter_config['neutral_weight'], classification_weighting=hyperparameter_config['cross_entropy_weighting'], contrastive=hyperparameter_config['contrastive'])
            test_loss, test_accuracy, test_neutral_accuracy, roc_auc = test(upuppi, test_loader, loss_fn= combined_classification_embedding_loss_puppi)
            print("Test losses: ", test_loss, test_accuracy, test_neutral_accuracy)
            # add in the losses and epoch to hyperparameter_dict
            hyperparameter_config['train_loss'], hyperparameter_config['test_loss'], hyperparameter_config['test_accuracy'], hyperparameter_config['test_neutral_accuracy'] = train_loss, test_loss, test_accuracy, test_neutral_accuracy
            hyperparameter_config['auc'] = roc_auc
            hyperparameter_config['epoch'] = epoch
            hyperparameter_list.append(copy.deepcopy(hyperparameter_config))
            # save the list
            with open('hyperparameter_list.txt', 'w') as f:
                f.write(str(hyperparameter_list))
            # check if the model has the best loss
            if roc_auc > best_hyperparameters['auc']:
                # update the best hyperparameters dict
                for key in hyperparameter_config:
                    best_hyperparameters[key] = hyperparameter_config[key]
                best_model = copy.deepcopy(upuppi)
                print("Best AUC: ", best_hyperparameters['auc'], "with hyperparameters: ", best_hyperparameters)
                # save the best model
                torch.save(best_model.state_dict(), os.path.join(home_dir, "best_model.pt"))
                # save the best model parameters
                with open("best_model_parameters.txt", "w") as f:
                    f.write(str(best_hyperparameters))

if __name__ == "__main__":
    hyperparameter_search()


