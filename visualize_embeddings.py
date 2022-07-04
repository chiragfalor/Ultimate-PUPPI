from helper_functions import *
import torch
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader

torch.manual_seed(40)

if __name__ == '__main__':
    data_test = UPuppiV0("/work/submit/cfalor/upuppi/Ultimate-PUPPI/test2/")
    model = "embedding_model"
    model = "GravNetConv"
    # model = "combined_model"
    # model = "combined_model2"
    model = "modelv2"
    # model = "modelv2_neg"
    # model = "modelv2_nz199"
    # model = "modelv2_nz0"
    # model = "modelv2_orig"
    model = "modelv2_contrastive"
    model = "modelv2_newdata"
    # model = "modelv3"
    # model = "Dynamic_GATv2"
    # model = "DynamicTransformer"
    # model = "DynamicPointTransformer"

    test_loader = DataLoader(data_test, batch_size=2, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
    model_dir = home_dir + 'models/{}/'.format(model)

    # model params
    epoch_num = 20
    net = get_neural_net(model)(pfc_input_dim=14)

    upuppi_state_dict = torch.load(model_dir + 'epoch-{}.pt'.format(epoch_num))['model']
    net.load_state_dict(upuppi_state_dict)
    net.eval()
    with torch.no_grad():
        data = next(iter(test_loader))

        pfc_truth = data.y.detach().numpy()
        vtx_truth = data.x_vtx[:, -1].detach().numpy()
        out, batch, pfc_embeddings, vtx_embeddings = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        # out, batch, pfc_embeddings = net(data.x_pfc, data.x_pfc_batch)

        # separate neutral and charged particles
        neutral_idx, charged_idx = torch.nonzero(data.x_pfc[:,-2] == 0).squeeze(), torch.nonzero(data.x_pfc[:,-2] != 0).squeeze()
        neutral_embeddings, charged_embeddings = pfc_embeddings[neutral_idx], pfc_embeddings[charged_idx]
        neutral_truth, charged_truth = pfc_truth[neutral_idx], pfc_truth[charged_idx]
        
        # plot the embeddings
        plot_2_embeddings(neutral_embeddings, charged_embeddings,'{}_{}_embeddings.png'.format(model, epoch_num), 
                            neutral_truth, charged_truth, "neutral", "charged")
        plot_2_embeddings(neutral_embeddings, charged_embeddings,'{}_{}_emb_color.png'.format(model, epoch_num), 
                            neutral_truth, charged_truth,"neutral", "charged", colored=True)