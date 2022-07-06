from helper_functions import *
import torch
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader

torch.manual_seed(41)

if __name__ == '__main__':
    data_test = UPuppiV0("/work/submit/cfalor/upuppi/Ultimate-PUPPI/test/")

    model_name = "vtx_pred_model_puppi"

    test_loader = DataLoader(data_test, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
    model_dir = home_dir + 'models/{}/'.format(model_name)

    # model params
    epoch_num = 19
    net = get_neural_net(model_name)()
    plot_against = 'z true'
    plot_against = 'vtx id'

    upuppi_state_dict = torch.load(model_dir + 'epoch-{:02d}.pt'.format(epoch_num))['model']
    net.load_state_dict(upuppi_state_dict)
    net.eval()
    with torch.no_grad():
        data = next(iter(test_loader))
        if plot_against == 'z true':
            pfc_truth = data.y.detach().numpy()
        elif plot_against == 'vtx id':
            pfc_truth = data.truth.detach().numpy()
            pfc_truth = (data.truth != 0).int().detach().numpy()
        else:
            raise ValueError("plot_against must be 'z' or 'vtx_id'")

        pfc_embeddings, vtx_embeddings = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        # out, batch, pfc_embeddings = net(data.x_pfc, data.x_pfc_batch)

        # separate neutral and charged particles
        neutral_idx, charged_idx = (data.x_pfc[:,-2] == 0), (data.x_pfc[:,-2] != 0)
        neutral_embeddings, charged_embeddings = pfc_embeddings[neutral_idx], pfc_embeddings[charged_idx]
        neutral_truth, charged_truth = pfc_truth[neutral_idx], pfc_truth[charged_idx]
        
        # plot the embeddings
        plot_2_embeddings(neutral_embeddings, charged_embeddings,'{}_{}_embeddings.png'.format(model_name, epoch_num), 
                            neutral_truth, charged_truth, "neutral", "charged")
        plot_2_embeddings(pfc_embeddings, vtx_embeddings, '{}_{}_vtx_pfc_emb.png'.format(model_name, epoch_num), pfc_truth, np.arange(len(vtx_embeddings)), "pfc", "vtx", colored = True)
        plot_2_embeddings(neutral_embeddings, charged_embeddings,'{}_{}_emb_color.png'.format(model_name, epoch_num), 
                            neutral_truth, charged_truth,"neutral", "charged", colored=True, colorbar_label=plot_against)