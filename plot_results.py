# reads the predicted z-value and actual z-value and plots the results
# data is stored in a csv file as dictionary at "/work/submit/cfalor/upuppi/z_reg/results/finalcsv.txt"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns

# model = "DynamicGCN"
model = "GravNetConv"
# model = "combined_model2"
# model = "combined_model"
model = "modelv2"
# model = "modelv3"
# model = "Dynamic_GATv2"
# df = pd.read_csv("/work/submit/cfalor/upuppi/z_reg/results/finalcsv.txt")
df = pd.read_csv("/work/submit/cfalor/upuppi/deepjet-geometric/results/{}.csv".format(model))
print(df.head())

# make 3 plots:
# 1. z-prediction vs. z-true
# 2. z-prediction vs. input_pt
# 3. z-prediction vs. input_eta
# on the same figure
fig, axs = plt.subplots(3, 1, figsize=(10,20))


# plot a color plot of zpred vs ztrue across the whole dataset
axs[0].scatter(df['ztrue'],df['zpred'],s=0.1,c='blue')
axs[0].set_xlabel('ztrue')
axs[0].set_ylabel('zpred')
axs[0].set_title('zpred vs ztrue across the whole dataset')
# plot a color plot of zpred vs ztrue only for particles with input_charge = 0
# print number of particles with input_charge = 0
print(df[df['charge'] == 0].shape)
axs[1].scatter(df[df['charge']==0]['ztrue'],df[df['charge']==0]['zpred'],s=0.1,c='blue')
axs[1].set_xlabel('ztrue')
axs[1].set_ylabel('zpred')
axs[1].set_title('zpred vs ztrue for particles with input_charge = 0')

# plot a color plot of zpred vs ztrue only for particles with input_charge != 0
# print number of particles with input_charge != 0
print(df[df['charge'] != 0].shape)
axs[2].scatter(df[df['charge']!=0]['ztrue'],df[df['charge']!=0]['zpred'],s=0.1,c='blue')
axs[2].set_xlabel('ztrue')
axs[2].set_ylabel('zpred')
axs[2].set_title('zpred vs ztrue for particles with input_charge != 0') 

plt.savefig('/work/submit/cfalor/upuppi/deepjet-geometric/results/{}_zpred_vs_ztrue.png'.format(model), bbox_inches='tight')
plt.close()

