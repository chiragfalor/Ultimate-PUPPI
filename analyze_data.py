from helper_functions import *

# seed the random number generator
torch.manual_seed(0)


data_train = UPuppiV0(home_dir + 'train5/')
data_test = UPuppiV0(home_dir + 'test5/')
train_loader = DataLoader(data_train, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])

data_params = {'event_num':[],'pfcs':[], 'vtxs':[], 'charges':[], 'neutrals':[], 'pileups':[] }
for counter, data in enumerate(tqdm(train_loader)):
    data_params['event_num'].append(counter)
    data_params['pfcs'].append(data.x_pfc.shape[0])
    data_params['vtxs'].append(data.x_vtx.shape[0])
    data_params['charges'].append((data.x_pfc[:,-2] != 0).sum().item())
    data_params['neutrals'].append((data.x_pfc[:,-2] == 0).sum().item())
    data_params['pileups'].append((data.truth == -1).sum().item())

df = pd.DataFrame(data_params)
print(df.describe())
print(df.head())
# plot histogram of pfc and vtx counts
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].hist(df['pfcs'], bins=100)
ax[1].hist(df['vtxs'], bins=100)
plt.savefig(home_dir + 'results/pfc_vtx_hist_2.png')
plt.close()

