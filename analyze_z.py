import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
from tqdm import tqdm


# get home directory path
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()

# BATCHSIZE = 64
data_train = UPuppiV0(home_dir + 'train/')
data_test = UPuppiV0(home_dir + 'test/')

# print(data_test)
data_loader = DataLoader(data_train, batch_size=3200, shuffle=True, follow_batch=['x_pfc'])

data = next(iter(data_loader))
# convert data.x_pfc to a pandas dataframe
x_pfc = data.x_pfc.numpy()
z = data.y.numpy()
x_pfc_df = pd.DataFrame(x_pfc)
x_pfc_df['z'] = z
# print summary of dataframe
# print(x_pfc_df.describe())
print(x_pfc_df.head())
# print(x_pfc_df.tail())
# print(x_pfc_df)

# select particles with pid = p
p = 6
x_pfc_df = x_pfc_df[x_pfc_df[p+4] == 1]
print(x_pfc_df.head())



# plot a histogram of z value of particles in the data
plt.hist(x_pfc_df['z'], bins=100)
# calculate the mean and standard deviation of z value of particles in the data
mean = x_pfc_df['z'].mean()
std = x_pfc_df['z'].std()
print('mean: ', mean)
print('std: ', std)
# plot the mean and standard deviation of z value of particles in the data
plt.axvline(x=mean, color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=mean + std, color='r', linestyle='dashed', linewidth=2)
plt.axvline(x=mean - std, color='r', linestyle='dashed', linewidth=2)
plt.savefig('z_hist_pid_{}.png'.format(13), bbox_inches='tight')



# x_pfc_df = pd.DataFrame(x_pfc)

