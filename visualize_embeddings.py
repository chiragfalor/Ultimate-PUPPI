from helper_functions import *
import torch
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader

torch.manual_seed(8)

if __name__ == '__main__':
    data_test = UPuppiV0(home_dir+"test_new/")

    model_name = "vtx_pred_model_puppi"
    model_name = 'multiclassifier_puppi_2_vtx'
    model_name = 'multiclassifier_puppi_without_primary'
    model_name = 'deep_multiclass_MET'
    model_name = "deep_multiclass_puppi"
    model_name = "cheat_model_try1"
    model_name = "deep_multiclass_no_emb_weight"
    # model_name = "deep_multiclass_neg_emb_weight"
    model_name = "multi_deep_new_data"
    model_name = "cheat_new_try2"


    test_loader = DataLoader(data_test, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
    model_dir = home_dir + 'models/{}/'.format(model_name)

    # model params
    epoch_num = 41
    net = get_neural_net(model_name)
    plot_against = 'z true'
    plot_against = 'vtx id'

    upuppi_state_dict = torch.load(model_dir + 'epoch-{:02d}.pt'.format(epoch_num))['model']
    net.load_state_dict(upuppi_state_dict)
    net.eval()
    with torch.no_grad():
        data = next(iter(test_loader))
        if plot_against == 'z true':
            pfc_truth = data.y.detach().numpy()
            # clamp z to be between -200 and 200
            pfc_truth[pfc_truth > 200] = 200
            pfc_truth[pfc_truth < -200] = -200
        elif plot_against == 'vtx id':
            pfc_truth = data.truth.int().detach().numpy()
            # clamp truth at 2
            pfc_truth[pfc_truth > 2] = 2
            pfc_truth[pfc_truth < 0] = 2
            # pfc_truth = (data.truth != 0).int().detach().numpy()
        else:
            raise ValueError("plot_against must be 'z' or 'vtx_id'")

        # pfc_embeddings, vtx_embeddings = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        # out, batch, pfc_embeddings = net(data.x_pfc, data.x_pfc_batch)
        out, pfc_embeddings, vtx_embeddings = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)

        # separate neutral and charged particles
        neutral_idx, charged_idx = (data.x_pfc[:,-2] == 0), (data.x_pfc[:,-2] != 0)
        neutral_embeddings, charged_embeddings = pfc_embeddings[neutral_idx], pfc_embeddings[charged_idx]
        neutral_truth, charged_truth = pfc_truth[neutral_idx], pfc_truth[charged_idx]
        
        # plot the embeddings
        plot_2_embeddings(neutral_embeddings, charged_embeddings,'{}/{}_embeddings.png'.format(model_name, epoch_num), 
                            neutral_truth, charged_truth, "neutral", "charged")
        plot_2_embeddings(pfc_embeddings, vtx_embeddings, '{}/{}_vtx_pfc_emb.png'.format(model_name, epoch_num), pfc_truth, np.arange(len(vtx_embeddings)), "pfc", "vtx", colored = True)
        plot_2_embeddings(neutral_embeddings, charged_embeddings,'{}/{}_emb_color.png'.format(model_name, epoch_num), 
                            neutral_truth, charged_truth,"neutral", "charged", colored=True, colorbar_label=plot_against)
        
        
        
        
        # # draw the embeddings of the original particles
        # px = data.x_pfc[:,0]
        # py = data.x_pfc[:,1]
        # pt = torch.sqrt(px**2 + py**2)
        # eta = data.x_pfc[:,2]
        # # eta = data.y
        # phi = torch.atan2(data.x_pfc[:,1], data.x_pfc[:,0])
        # eta_phi = torch.cat((eta.unsqueeze(1), phi.unsqueeze(1)), 1)
        # px_py = torch.cat((px.unsqueeze(1), py.unsqueeze(1)), 1)
        # charged_eta_phi = eta_phi[charged_idx]
        # neutral_eta_phi = eta_phi[neutral_idx]
        # charged_px_py = px_py[charged_idx]
        # neutral_px_py = px_py[neutral_idx]
        # charged_pt = pt[charged_idx]
        # neutral_pt = pt[neutral_idx]
        # # plot eta and phi
        # plt.scatter(charged_eta_phi[:,0].numpy(), charged_eta_phi[:,1].numpy(), marker='*', c=charged_truth, cmap='jet')
        # plt.scatter(neutral_eta_phi[:,0].numpy(), neutral_eta_phi[:,1].numpy(), marker='.', c=neutral_truth, cmap='jet')
        # # label
        # plt.xlabel('$\eta$')
        # plt.ylabel('$\phi$')
        # plt.savefig(home_dir + 'results/{}/{}_eta_phi.png'.format(model_name, epoch_num), bbox_inches='tight')
        # plt.close()
        # # plot px and py
        # plt.scatter(charged_px_py[:,0].numpy(), charged_px_py[:,1].numpy(), marker='*', c=charged_truth, cmap='jet')
        # plt.scatter(neutral_px_py[:,0].numpy(), neutral_px_py[:,1].numpy(), marker='.', c=neutral_truth, cmap='jet')
        # # label
        # plt.xlabel('$p_x$')
        # plt.ylabel('$p_y$')
        # plt.savefig(home_dir + 'results/{}/{}_px_py.png'.format(model_name, epoch_num), bbox_inches='tight')
        # plt.close()
        # # plot pt and phi
        # plt.scatter(charged_pt.numpy(), charged_eta_phi[:,1].numpy(), marker='*', c=charged_truth, cmap='jet')
        # plt.scatter(neutral_pt.numpy(), neutral_eta_phi[:,1].numpy(), marker='.', c=neutral_truth, cmap='jet')
        # # label
        # plt.xlabel('$p_T$')
        # plt.ylabel('$\phi$')
        # plt.savefig(home_dir + 'results/{}/{}_pt_phi.png'.format(model_name, epoch_num), bbox_inches='tight')
        # # plot_2_embeddings(neutral_eta_phi, charged_eta_phi, '{}/{}_emb_color_orig.png'.format(model_name, epoch_num),
        # #                     neutral_truth, charged_truth, "neutral", "charged", colored=True, colorbar_label=plot_against)
