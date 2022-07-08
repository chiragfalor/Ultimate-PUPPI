from helper_functions import *

BATCHSIZE = 32
if __name__ == "__main__":
    epoch_to_load = 39
    model_name = "DynamicPointTransformer"
    # model_name = 'modelv2_analysis'
    model_name = 'modelv3_first_try'
    model_name = 'pileup_classifier_puppi'
    model_name = 'multiclassifier_puppi'
    model_name = 'multiclassifier_puppi_2_vtx'
    # model_name = 'multiclassifier_puppi_no_concat'
    # model_name = 'multiclassifier_puppi_with_concat_z'

    # model_name = 'multiclassifier_2_vtx_embloss'
    model_name = 'multiclassifier_puppi_2_vtx_weighted'
    # model_name = 'multiclassifier_puppi_5_vtx_weighted'


    net = get_neural_net(model_name)(dropout=0, vtx_classes=2)

    data_test = UPuppiV0(home_dir + "test/")
    test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                            follow_batch=['x_pfc', 'x_vtx'])

    model_dir = home_dir + 'models/{}/'.format(model_name)
    model_loc = os.path.join(model_dir, 'epoch-{:02d}.pt'.format(epoch_to_load))
    print("Saving/Plotting predictions of model {}".format(model_name), "at epoch {}".format(epoch_to_load))

    model_state_dict = torch.load(model_loc)['model']
    net.load_state_dict(model_state_dict)


save_name = '{}/epoch-{:02d}'.format(model_name, epoch_to_load)
if not os.path.exists(home_dir+'results/'+model_name): os.makedirs(home_dir+'results/'+model_name)




# # get df by uncommenting one of the following lines
# df = save_z_predictions(net, test_loader, save_name)
df = save_class_predictions(net, test_loader, save_name)
df = pd.read_csv(home_dir + 'results/{}.csv'.format(save_name))
# select particles with vtx_truth == 0
# df = df[df['vtx_truth'] != 0]
print(df.head())
# plot_z_pred_z_true(df, save_name)

plot_multiclassification_metrics(df, save_name)
