from helper_functions import *

# seed the random number generator
torch.manual_seed(0)


data_train = UPuppiV0(home_dir + 'train/')
data_test = UPuppiV0(home_dir + 'test/')
train_loader = DataLoader(data_train, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])

data_params = {'event_num':[], 'vtx0_px':[], 'vtx1_px':[], 'vtx2_px':[], 'vtx0_py':[], 'vtx1_py':[], 'vtx2_py':[], 'vtx0_count':[], 'vtx1_count':[], 'vtx2_count':[]}
for counter, data in enumerate(tqdm(train_loader)):
    data_params['event_num'].append(counter)
    px = data.x_pfc[:,0]
    py = data.x_pfc[:,1]
    vtx0_id = (data.truth == 0)
    vtx1_id = (data.truth == 1)
    vtx2_id = (data.truth == 2)
    data_params['vtx0_px'].append(px[vtx0_id].sum().item())
    data_params['vtx1_px'].append(px[vtx1_id].sum().item())
    data_params['vtx2_px'].append(px[vtx2_id].sum().item())
    data_params['vtx0_py'].append(py[vtx0_id].sum().item())
    data_params['vtx1_py'].append(py[vtx1_id].sum().item())
    data_params['vtx2_py'].append(py[vtx2_id].sum().item())
    data_params['vtx0_count'].append(vtx0_id.sum().item())
    data_params['vtx1_count'].append(vtx1_id.sum().item())
    data_params['vtx2_count'].append(vtx2_id.sum().item())



df = pd.DataFrame(data_params)
print(df.describe())
print(df.head())
# plot histogram of px and py for each vtx
fig, ax = plt.subplots(1,3, figsize=(10,5))
ax[0].hist(df['vtx0_px'], bins=100)
ax[1].hist(df['vtx1_px'], bins=100)
ax[2].hist(df['vtx2_px'], bins=100)
# label and title
ax[0].set_title('px for vtx0')
ax[1].set_title('px for vtx1')
ax[2].set_title('px for vtx2')
ax[0].set_xlabel('px')
ax[1].set_xlabel('px')
ax[2].set_xlabel('px')
ax[0].set_ylabel('count')
ax[1].set_ylabel('count')
ax[2].set_ylabel('count')
plt.tight_layout()
plt.savefig(home_dir + 'results/vtx_px_hist.png', bbox_inches='tight')
plt.close()
fig, ax = plt.subplots(1,3, figsize=(10,5))
ax[0].hist(df['vtx0_py'], bins=100)
ax[1].hist(df['vtx1_py'], bins=100)
ax[2].hist(df['vtx2_py'], bins=100)
# label and title
ax[0].set_title('py for vtx0')
ax[1].set_title('py for vtx1')
ax[2].set_title('py for vtx2')
ax[0].set_xlabel('py')
ax[1].set_xlabel('py')
ax[2].set_xlabel('py')
ax[0].set_ylabel('count')
ax[1].set_ylabel('count')
ax[2].set_ylabel('count')
plt.tight_layout()

plt.savefig(home_dir + 'results/vtx_py_hist.png', bbox_inches='tight')
plt.close()

# plot histogram of vtx count for each vtx
fig, ax = plt.subplots(1,3, figsize=(10,5))
# plot bincounts for each vtx
ax[0].hist(df['vtx0_count'], bins=np.arange(0,200,1))
ax[1].hist(df['vtx1_count'], bins=np.arange(0,150,1))
ax[2].hist(df['vtx2_count'], bins=np.arange(0,100,1))

# label and title
ax[0].set_title('pfc count for vtx0')
ax[1].set_title('pfc count for vtx1')
ax[2].set_title('pfc count for vtx2')
plt.savefig(home_dir + 'results/vtx_count_hist.png', bbox_inches='tight')
plt.close()
