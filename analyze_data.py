from helper_functions import *
# from dataset_graph_loader import UPuppiV0
# from dense_graph_loader import UPuppiV0

# seed the random number generator
torch.manual_seed(0)

data_train = UPuppiV0(home_dir + "all_new_data/")
# data_train = UPuppiV0(home_dir + 'train5/')
data_test = UPuppiV0(home_dir + 'test5/')
train_loader = DataLoader(data_train, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])

data_params = {'event_num':[],'pfcs':[], 'vtxs':[], 'charges':[], 'neutrals':[], 'pileups':[], 'vtx_pt':[], 'vtx0_pt':[], 'vtx0_eta_max':[], 'vtx0_eta_min':[]}
for counter, data in enumerate(tqdm(train_loader)):
    if data.x_vtx.shape[0] == 0:
        continue
    vtx0_id = (data.truth == 0)
    data_params['event_num'].append(counter)
    data_params['pfcs'].append(data.x_pfc.shape[0])
    data_params['vtxs'].append(data.x_vtx.shape[0])
    data_params['charges'].append((data.x_pfc[:,-2] != 0).sum().item())
    data_params['neutrals'].append((data.x_pfc[:,-2] == 0).sum().item())
    data_params['pileups'].append((data.truth == -1).sum().item())
    data_params['vtx_pt'].append(data.x_vtx[:, -1].sum().item())
    data_params['vtx0_pt'].append(data.x_vtx[0, -1].item())
    try:
        data_params['vtx0_eta_max'].append(data.x_pfc[vtx0_id, 2].max().item())
        data_params['vtx0_eta_min'].append(data.x_pfc[vtx0_id, 2].min().item())
    except:
        print(vtx0_id.sum().item())
        print(data.x_vtx[0,:])
        print(data.truth)
        print(data.x_vtx.shape)


df = pd.DataFrame(data_params)
# select events where vtx0_pt > 100
# df = df[df.vtx0_pt > 100]
print(df.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
print(df.head())
# plot histogram of pfc and vtx counts
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].hist(df['pfcs'], bins=100)
ax[1].hist(df['vtxs'], bins=100)
plt.savefig(home_dir + 'results/pfc_vtx_hist_2.png', bbox_inches='tight')
plt.close()

# plot histogram of vtx pt
fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.hist(df['vtx_pt'], bins=100)
plt.savefig(home_dir + 'results/vtx_pt_hist_2.png', bbox_inches='tight')
plt.close()

# plot histogram of vtx0 pt
fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.hist(df['vtx0_pt'], bins=100)
plt.savefig(home_dir + 'results/vtx0_pt_hist_2.png', bbox_inches='tight')
plt.close()

# plot histogram of vtx0 eta
fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.hist(df['vtx0_eta'], bins=100)
plt.savefig(home_dir + 'results/vtx0_eta_hist_2.png', bbox_inches='tight')
plt.close()


