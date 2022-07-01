from helper_functions import *

BATCHSIZE = 32
epoch_to_load = 1
model_name = "DynamicPointTransformer"

net = get_neural_net(model_name)(pfc_input_dim=13, k1=32, k2=8, dropout=0)

data_test = UPuppiV0(home_dir + "test/")
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                        follow_batch=['x_pfc', 'x_vtx'])

model_dir = home_dir + 'models/{}/'.format(model_name)
model_loc = os.path.join(model_dir, 'epoch-{:02d}.pt'.format(epoch_to_load))
print("Saving/Plotting predictions of model {}".format(model_name), "at epoch {}".format(epoch_to_load))

model_state_dict = torch.load(model_loc)['model']
net.load_state_dict(model_state_dict)
save_name = '{}/{}_epoch-{:02d}'.format(model_name, model_name, epoch_to_load)

# # get df by uncommenting one of the following lines
# df = save_predictions(net, test_loader, save_name)
df = pd.read_csv(home_dir + 'results/{}.csv'.format(save_name))
plot_z_pred_z_true(df, save_name)