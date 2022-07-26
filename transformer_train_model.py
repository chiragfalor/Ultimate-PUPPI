import time
from helper_functions import *
from loss_functions import *
from dataset_graph_loader import UPuppiV0

start_time = time.time()

data_train = UPuppiV0(home_dir + 'train9/')
data_test = UPuppiV0(home_dir + 'test9/')
BATCHSIZE = 64

model_name = 'new_transformer_try1'
model_name = 'simple_transformer_try1'

vtx_classes = 1


print("Training {}...".format(model_name))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("Using device: ", device, torch.cuda.get_device_name(0))

model_dir = home_dir + 'models/{}/'.format(model_name)
net = get_neural_net(model_name, new_net=True)(dropout=0, vtx_classes=vtx_classes).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# save the model hyperparameters in the model directory
if not os.path.exists(model_dir): os.makedirs(model_dir)
with open(model_dir + 'hyperparameters.txt', 'w') as f: 
    f.write("network_architecture: {}\n".format(net))
    f.write("optimizer: {}\n".format(optimizer))
# save param dict
with open(model_dir + 'param_dict.pkl', 'wb') as f: 
    param_dict = net.get_param_dict()
    pickle.dump(param_dict, f)


train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])



def train(model, optimizer, loss_fn, neutral_weight = 1, classification_weighting = 'pt'):
    '''
    Trains the given model for one epoch
    '''
    global vtx_classes
    model.train()
    train_loss = 0
    for counter, data in enumerate(tqdm(train_loader)):
        data = process_data(data)
        data.to(device)
        optimizer.zero_grad()
        neutral_mask = data.x_pfc[:, -2] == 0
        pt = torch.sqrt(data.x_pfc[:, 0]**2 + data.x_pfc[:, 1]**2).float()
        scores = model(data.x_pfc, data.edge_index)
        truth = process_truth(data.truth, vtx_classes).long()
        loss = loss_fn(scores, truth, neutral_mask, neutral_weight, weighting=classification_weighting, pt=pt)
        # if loss is nan, print everything
        if np.isnan(loss.item()):
            raise(Exception("Loss is nan"))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if counter % 100 == 1:
            print("Train loss: ", train_loss / counter)
            print("Current loss: ", loss.item())
    return train_loss / counter

@torch.no_grad()
def test(model, loss_fn):
    '''
    Tests the given model on the test set
    '''
    global epoch, vtx_classes
    model.eval()
    test_accuracy, test_neutral_accuracy, test_loss = 0, 0, 0
    all_pred_prob = None
    neutral_pred_prob = None
    for counter, data in enumerate(tqdm(test_loader)):
        data = process_data(data)
        data.to(device)
        score = model(data.x_pfc, data.edge_index)
        vtx_classes = score.shape[1] - 1
        loss = loss_fn(score, process_truth(data.truth, vtx_classes).long())
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
    if vtx_classes > 1:
        neutral_truth = (neutral_truth != 0).long()
    fpr, tpr, thresholds = metrics.roc_curve(neutral_truth.cpu().numpy(), 1 - neutral_pred_prob.cpu().numpy()[:, 0])
    roc_auc = metrics.auc(fpr, tpr)
    # plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # add title and save
    plt.title('Neutral AUC score: {:.2f}'.format(roc_auc))
    plt.savefig(model_dir + '/epoch_{:02d}.png'.format(epoch), bbox_inches='tight')
    plt.close()
    return test_loss / counter, test_accuracy / counter, test_neutral_accuracy / counter, roc_auc


NUM_EPOCHS = 20

model_performance = []

for epoch in range(NUM_EPOCHS):
    train_loss = train(net, optimizer, loss_fn=multiclassification_loss, neutral_weight=epoch+1, classification_weighting='pt')
    state_dicts = {'model':net.state_dict(),
                    'opt':optimizer.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{:02d}.pt'.format(epoch)))
    print("Model saved")
    print("Time elapsed: ", time.time() - start_time)
    print("-----------------------------------------------------")
    test_loss, test_accuracy, test_neutral_accuracy, auc = test(net, multiclassification_loss)
    print("Epoch: {:02d}, Train Loss: {:4f}, Test Loss: {:4f} Test Accuracy: {:.2%}, Test Neutral Accuracy: {:.2%}, AUC: {:.2f}".format(epoch, train_loss, test_loss, test_accuracy, test_neutral_accuracy, auc))
    model_performance.append({'epoch':epoch, 'train_loss':train_loss, 'test_accuracy':test_accuracy, 'test_neutral_accuracy':test_neutral_accuracy, 'test_loss':test_loss, 'auc':auc})
# save the model performance as txt
with open(model_dir + 'model_performance.txt', 'w') as f:
    for item in model_performance:
        f.write("{}\n".format(item))

epoch_to_load = NUM_EPOCHS - 1

save_name = '{}/epoch-{:02d}'.format(model_name, epoch_to_load)
if not os.path.exists(home_dir+'results/'+model_name): os.makedirs(home_dir+'results/'+model_name)
df = save_class_predictions(net, test_loader, save_name)
plot_multiclassification_metrics(df, save_name)
