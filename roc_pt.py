from helper_functions import *

data_train = UPuppiV0(home_dir + 'train5/')
data_test = UPuppiV0(home_dir + 'test5/')
BATCHSIZE = 1
train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=False, num_workers=0)

pt = torch.tensor([])
truth = torch.tensor([])
charge = torch.tensor([])
E = torch.tensor([])
eta = torch.tensor([])
for data in tqdm(train_loader):
    pt = torch.cat((pt, (data.x_pfc[:, 0]**2 + data.x_pfc[:, 1]**2)**0.5), 0)
    truth = torch.cat((truth, (data.truth == 0).float()), 0)
    charge = torch.cat((charge, data.x_pfc[:, -2]), 0)
    E = torch.cat((E, data.x_pfc[:, 3]), 0)
    eta = torch.cat((eta, data.x_pfc[:, 2]), 0)
    

# neutral_mask = (charge == 0)
# pt = pt[neutral_mask]
# truth = truth[neutral_mask]
# E = E[neutral_mask]
# convert to numpy
pt = pt.numpy()
truth = truth.numpy()
E = E.numpy()
# construct a roc curve
fpr, tpr, _ = metrics.roc_curve(truth, E)
roc_auc = metrics.auc(fpr, tpr)
print("ROC AUC: ", roc_auc)
# plot the roc curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# add title and save
plt.title('AUC score: {:.2f}'.format(roc_auc))
plt.savefig('E_roc.png', bbox_inches='tight')
plt.close()
