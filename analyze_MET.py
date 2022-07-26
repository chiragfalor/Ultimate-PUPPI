from helper_functions import *

# seed the random number generator
torch.manual_seed(0)


data_train = UPuppiV0(home_dir + 'train/')
data_test = UPuppiV0(home_dir + 'test/')
train_loader = DataLoader(data_train, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])

data_params = {'event_num':[], 'vtx0_px':[], 'vtx1_px':[], 'vtx2_px':[], 'vtx0_py':[], 'vtx1_py':[], 'vtx2_py':[], 'vtx0_count':[], 'vtx1_count':[], 'vtx2_count':[], 'vtx0_pt':[], 'vtx1_pt':[], 'vtx0_charged_pt':[], 'vtx1_charged_pt':[], 'vtx2_charged_pt':[], 'vtx0_neutral_pt':[], 'vtx1_neutral_pt':[], 'vtx2_neutral_pt':[]}
vtx0_pfc_params = {'px':[], 'py':[], 'pt':[], 'eta':[], 'phi':[], 'charge':[]}
vtx1_pfc_params = {'px':[], 'py':[], 'pt':[], 'eta':[], 'phi':[], 'charge':[]}
for counter, data in enumerate(tqdm(train_loader)):
    data_params['event_num'].append(counter)
    px = data.x_pfc[:,0]
    py = data.x_pfc[:,1]
    pt = (px**2 + py**2)**0.5
    eta = data.x_pfc[:,2]
    # arctan py/px to get phi
    phi = np.arctan2(py, px)
    vtx0_id = (data.truth == 0)
    vtx1_id = (data.truth == 1)
    vtx2_id = (data.truth == 2)
    charged_mask = (data.x_pfc[:,-2] != 0)
    eta_charged_mask = (eta>3) & charged_mask
    data_params['vtx0_px'].append(px[vtx0_id].sum().item())
    data_params['vtx1_px'].append(px[vtx1_id].sum().item())
    data_params['vtx2_px'].append(px[vtx2_id].sum().item())
    data_params['vtx0_py'].append(py[vtx0_id].sum().item())
    data_params['vtx1_py'].append(py[vtx1_id].sum().item())
    data_params['vtx2_py'].append(py[vtx2_id].sum().item())
    data_params['vtx0_count'].append(vtx0_id.sum().item())
    data_params['vtx1_count'].append(vtx1_id.sum().item())
    data_params['vtx2_count'].append(vtx2_id.sum().item())
    data_params['vtx0_pt'].append(pt[vtx0_id].sum().item())
    data_params['vtx1_pt'].append(pt[vtx1_id].sum().item())
    data_params['vtx0_charged_pt'].append(pt[charged_mask & vtx0_id].sum().item())
    data_params['vtx1_charged_pt'].append(pt[charged_mask & vtx1_id].sum().item())
    data_params['vtx2_charged_pt'].append(pt[charged_mask & vtx2_id].sum().item())
    data_params['vtx0_neutral_pt'].append(pt[~charged_mask & vtx0_id].sum().item())
    data_params['vtx1_neutral_pt'].append(pt[~charged_mask & vtx1_id].sum().item())
    data_params['vtx2_neutral_pt'].append(pt[~charged_mask & vtx2_id].sum().item())
    vtx0_pfc_params['px'].extend(px[vtx0_id].squeeze().tolist())
    vtx0_pfc_params['py'].extend(py[vtx0_id].squeeze().tolist())
    vtx0_pfc_params['pt'].extend(pt[vtx0_id].squeeze().tolist())
    vtx0_pfc_params['eta'].extend(eta[vtx0_id].squeeze().tolist())
    vtx0_pfc_params['phi'].extend(phi[vtx0_id].squeeze().tolist())
    vtx0_pfc_params['charge'].extend(data.x_pfc[vtx0_id,-2].squeeze().tolist())
    vtx1_pfc_params['px'].extend(px[vtx1_id].squeeze().tolist())
    vtx1_pfc_params['py'].extend(py[vtx1_id].squeeze().tolist())
    vtx1_pfc_params['pt'].extend(pt[vtx1_id].squeeze().tolist())
    vtx1_pfc_params['eta'].extend(eta[vtx1_id].squeeze().tolist())
    vtx1_pfc_params['phi'].extend(phi[vtx1_id].squeeze().tolist())
    vtx1_pfc_params['charge'].extend(data.x_pfc[vtx1_id,-2].squeeze().tolist())
    # if any eta charged is True, then the event is a signal event
    if any(eta_charged_mask):
        print(data.x_pfc[eta_charged_mask,:])
        



    





df = pd.DataFrame(data_params)
vtx0_pfc_df = pd.DataFrame(vtx0_pfc_params)
vtx1_pfc_df = pd.DataFrame(vtx1_pfc_params)
print(vtx0_pfc_df.head())
print(vtx1_pfc_df.head())
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

# plot histogram of total pt for each vtx. Red for vtx0, blue for vtx1
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.hist(df['vtx0_pt'], bins=200, color='red', label='vtx0', alpha=0.5)
ax.hist(df['vtx1_pt'], bins=200, color='blue', label='vtx1', alpha=0.5)
plt.legend()
plt.savefig(home_dir + 'results/vtx_pt_hist.png', bbox_inches='tight')
plt.close()

# plot charged pt for each vtx. Red for vtx0, blue for vtx1
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.hist(df['vtx0_charged_pt'], bins=200, color='red', label='vtx0', alpha=0.5)
ax.hist(df['vtx1_charged_pt'], bins=200, color='green', label='vtx1', alpha=0.5)
ax.hist(df['vtx2_charged_pt'], bins=200, color='blue', label='vtx2', alpha=0.5)
plt.legend()
plt.savefig(home_dir + 'results/vtx_charged_pt_hist.png', bbox_inches='tight')
plt.close()

# plot neutral pt for each vtx. Red for vtx0, blue for vtx1
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.hist(df['vtx0_neutral_pt'], bins=np.linspace(0,800,100), color='red', label='vtx0', alpha=0.5)
ax.hist(df['vtx1_neutral_pt'], bins=np.linspace(0,800,100), color='blue', label='vtx1', alpha=0.5)
ax.hist(df['vtx2_neutral_pt'], bins=np.linspace(0,800,100), color='green', label='vtx2', alpha=0.5)
plt.legend()
plt.savefig(home_dir + 'results/vtx_neutral_pt_hist.png', bbox_inches='tight')
plt.close()

# plot charged vs neutral pt for vtx0. Red for charged pt, blue for neutral pt
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.hist(df['vtx0_charged_pt'], bins=np.arange(0,800,20), color='red', label='charged pt', alpha=0.5)
ax.hist(df['vtx0_neutral_pt'], bins=np.arange(0,800,20), color='blue', label='neutral pt', alpha=0.5)
plt.legend()
plt.savefig(home_dir + 'results/vtx0_charged_vs_neutral_pt_hist.png', bbox_inches='tight')
plt.close()

# plot charged vs neutral pt for vtx1. Red for charged pt, blue for neutral pt
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.hist(df['vtx1_charged_pt'], bins=np.linspace(0,400,100), color='red', label='charged pt', alpha=0.5)
ax.hist(df['vtx1_neutral_pt'], bins=np.linspace(0,400,100), color='blue', label='neutral pt', alpha=0.5)
plt.legend()
plt.savefig(home_dir + 'results/vtx1_charged_vs_neutral_pt_hist.png', bbox_inches='tight')
plt.close()

# plot charged vs neutral eta for vtx0. Red for charged eta, blue for neutral eta
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.hist(vtx0_pfc_df[vtx0_pfc_df['charge']!=0]['eta'], bins=np.linspace(-5, 5, 30), color='red', label='charged eta', alpha=0.5)
ax.hist(vtx0_pfc_df[vtx0_pfc_df['charge']==0]['eta'], bins=np.linspace(-5, 5, 30), color='blue', label='neutral eta', alpha=0.5)
plt.legend()
plt.savefig(home_dir + 'results/vtx0_charged_vs_neutral_eta_hist.png', bbox_inches='tight')
plt.close()

# plot charged vs neutral eta for vtx1. Red for charged eta, blue for neutral eta
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.hist(vtx1_pfc_df[vtx1_pfc_df['charge']!=0]['eta'], bins=np.linspace(-5, 5, 30), color='red', label='charged eta', alpha=0.5)
ax.hist(vtx1_pfc_df[vtx1_pfc_df['charge']==0]['eta'], bins=np.linspace(-5, 5, 30), color='blue', label='neutral eta', alpha=0.5)
plt.legend()
plt.savefig(home_dir + 'results/vtx1_charged_vs_neutral_eta_hist.png', bbox_inches='tight')
plt.close()

# plot eta for each vtx. Red for vtx0, blue for vtx1
fig, ax = plt.subplots(1,1, figsize=(10,10))
ax.hist(vtx0_pfc_df['eta'], bins=np.linspace(-5, 5, 30), color='red', label='vtx0', alpha=0.5)
ax.hist(vtx1_pfc_df['eta'], bins=np.linspace(-5, 5, 30), color='blue', label='vtx1', alpha=0.5)
plt.legend()
plt.savefig(home_dir + 'results/vtx_eta_hist.png', bbox_inches='tight')
plt.close()







# df describe for pt
print(df['vtx0_pt'].describe())
print(df['vtx1_pt'].describe())
