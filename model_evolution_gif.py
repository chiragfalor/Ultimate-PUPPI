# loads model, predicts z, and plots z_pred vs z_true for all epochs. Makes a gif of the predictions.
from helper_functions import *



if __name__ == '__main__':
    model_name = 'DynamicPointTransformer'
    model_name = 'modelv2_analysis'
    model_name = 'modelv2'
    net = get_neural_net(model_name)(pfc_input_dim=13, hidden_dim=256, k1=32, k2=16, dropout=0)

    data_test = UPuppiV0(home_dir + "test/")
    test_loader = DataLoader(data_test, batch_size=32, shuffle=True,
                        follow_batch=['x_pfc', 'x_vtx'])
    
    make_model_evolution_gif(net, model_name, test_loader)


