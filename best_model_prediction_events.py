from helper_functions import *
from loss_functions import *

import h5py

def choose_nice_events(net, data_loader, loss_cut = 0.88):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")
    net.to(device)
    net.eval()
    
    pfs = torch.empty((0, 1000, 12))
    vtxs = torch.empty((0, 200, 5))
    truth = torch.empty((0, 1000))
    z = torch.empty((0, 1000))

    counter = 0
    with torch.no_grad():
        for data in tqdm(data_loader):
            data = process_data(data)
            data = data.to(device)
            scores, _, _ = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
            vtx_classes = scores.shape[1] - 1
            class_true_batch = process_truth(data.truth, vtx_classes).long()
            neutral_mask = (data.x_pfc[:, -2]==0)
            loss = multiclassification_loss(scores[neutral_mask], class_true_batch[neutral_mask], weighting='num')
            if loss > loss_cut:
                continue
            else:
                data = data.to(cpu)
                counter += 1
                pfs_event = torch.cat((data.x_pfc, torch.zeros((1000 - data.x_pfc.shape[0], data.x_pfc.shape[1]))), dim=0).unsqueeze(0)
                vtx_event = torch.cat((data.x_vtx, torch.zeros((200 - data.x_vtx.shape[0], data.x_vtx.shape[1]))), dim=0).unsqueeze(0)
                truth_event = torch.cat((data.truth, torch.zeros((1000 - data.truth.shape[0],))), dim=0).unsqueeze(0)
                z_event = torch.cat((data.y, torch.zeros((1000 - data.y.shape[0],))), dim=0).unsqueeze(0)
                pfs = torch.cat((pfs, pfs_event), dim=0)
                vtxs = torch.cat((vtxs, vtx_event), dim=0)
                truth = torch.cat((truth, truth_event), dim=0)
                z = torch.cat((z, z_event), dim=0)
                if counter % 1000 == 0 and counter != 0:
                    print("Saving data")
                    save_path = home_dir+ 'train6/raw/samples_v0_dijet_'+str(50 + counter//1000)+".h5"
                    file_out = h5py.File(save_path , "w")
                    file_out.create_dataset("pfs", data=pfs.numpy())
                    file_out.create_dataset("vtx", data=vtxs.numpy())
                    file_out.create_dataset("truth", data=truth.numpy())
                    file_out.create_dataset("z", data=z.numpy())
                    file_out.create_dataset("n", data=pfs.shape[0], dtype='i8')
                    file_out.close()
                    print("Saved data to {}".format(save_path))
                    print("Processed {} events".format(counter))
                    pfs = torch.empty((0, 1000, 12))
                    vtxs = torch.empty((0, 200, 5))
                    truth = torch.empty((0, 1000))
                    z = torch.empty((0, 1000))
            






def save_class_predictions(net, data_loader, save_name, pt_cut=650, E_cut=750, loss_cut = 0.5):
    '''
    Saves the data and predictions of the network for all the data in the data_loader
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    total_loss = 0
    # initialize np arrays to save the data and predictions
    class_prob = None
    for data in tqdm(data_loader):
        data = process_data(data)
        total_pt = (data.x_pfc[:, 0]**2 + data.x_pfc[:, 1]**2).sqrt().sum()
        total_E = data.x_pfc[:, 3].sum()
        if total_pt.item() < pt_cut or total_E.item() < E_cut:
            continue
        data = data.to(device)
        with torch.no_grad():
            scores_batch, pfc_emb, vtx_emb = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
            vtx_classes = scores_batch.shape[1] - 1
            class_prob_batch = nn.Softmax(dim=1)(scores_batch)
            class_true_batch = process_truth(data.truth, vtx_classes).long()
            pred_batch = torch.argmax(class_prob_batch, dim=1)
            neutral_mask = (data.x_pfc[:, -2]==0)
            loss = multiclassification_loss(scores_batch[neutral_mask], class_true_batch[neutral_mask], weighting='num')
            if loss > loss_cut:
                continue
            if class_prob is None:
                class_prob = class_prob_batch
                class_true = class_true_batch
                pred = pred_batch
                vtx_truth = data.truth
                scores = scores_batch
                charge = data.x_pfc[:, -2]
            else:
                class_prob = torch.cat((class_prob, class_prob_batch), dim=0)
                class_true = torch.cat((class_true, class_true_batch), dim=0)
                pred = torch.cat((pred, pred_batch), dim=0)
                vtx_truth = torch.cat((vtx_truth, data.truth), dim=0)
                scores = torch.cat((scores, scores_batch), dim=0)
                charge = torch.cat((charge, data.x_pfc[:, -2]), dim=0)
    # cross entropy loss
    total_loss = nn.CrossEntropyLoss()(scores, class_true)
    accuracy = torch.mean((pred == class_true).float())
    print("Total loss: {}".format(total_loss), " and accuracy: {:.2%} with {} vertex classes".format(accuracy, vtx_classes))
    data_dict = {'class_true': class_true, 'charge': charge, 'pred': pred, 'vtx_truth': vtx_truth}
    # add the class predictions to the data_dict
    for i in range(class_prob.shape[1]):
        data_dict['class_prob_{}'.format(i)] = class_prob[:, i]
    # convert all tensors to numpy to be able to save dataframe
    for key in data_dict:
        data_dict[key] = data_dict[key].detach().cpu().numpy()
    df = pd.DataFrame(data_dict)
    df.to_csv(home_dir + 'results/{}.csv'.format(save_name), index=False)
    print("Saved data and predictions to results/{}.csv".format(save_name))
    return df


def save_event_predictions(net, test_loader, save_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net.to(device)
    net.eval()
    df = pd.DataFrame()
    for i, data in enumerate(tqdm(test_loader)):
        data = process_data(data)
        data = data.to(device)
        with torch.no_grad():
            scores, _, _ = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
            vtx_classes = scores.shape[1] - 1
            class_prob = scores.softmax(dim=1)
            class_true = process_truth(data.truth, vtx_classes).long()
            total_pt = (data.x_pfc[:, 0]**2 + data.x_pfc[:, 1]**2).sqrt().sum()
            total_E = data.x_pfc[:, 3].sum()
            loss = multiclassification_loss(scores, class_true, weighting='num')
            # get loss among neutral particles
            neutral_mask = (data.x_pfc[:, -2]==0)
            neutral_loss = multiclassification_loss(scores[neutral_mask], class_true[neutral_mask], weighting='num')
            neutral_mask = neutral_mask.cpu().numpy()
            truth = (class_true == 0).int().detach().cpu().numpy()
            # get auc in neutral particles
            fpr, tpr, _ = metrics.roc_curve(truth[neutral_mask], class_prob[neutral_mask, 0].cpu().numpy())
            roc_auc = metrics.auc(fpr, tpr)

            # save indexing by event_num, loss, auc, and loss in neutral particles, total pt and E
            df = df.append(pd.DataFrame({'event_num': i, 'loss': loss.item(), 'loss_neutral': neutral_loss.item(), 'auc': roc_auc, 'total_pt': total_pt.item(), 'total_E': total_E.item()}, index=[i]))
            # save df to csv
    # drop the unnamed index
    df.reset_index(drop=True, inplace=True)
    # print columns
    print(df.columns)

    # df.drop(["Unnamed: 0"], axis=1, inplace=True)
    df.to_csv(home_dir+'results/'+save_name+'.csv', index=False)
    print("Saved data to results/{}.csv".format(save_name))
    return df



if __name__ == "__main__":
    save = True
    process = False
    epoch_to_load = 99
    model_name = 'deep_multiclass_test'
    # model_name = 'deep_multiclass_enhanced_data_try2'

    if save:
        net = get_neural_net(model_name)

        data_test = UPuppiV0(home_dir + "test6/")
        test_loader = DataLoader(data_test, batch_size=1, shuffle=True,
                                    follow_batch=['x_pfc', 'x_vtx'])

        model_dir = home_dir + 'models/{}/'.format(model_name)
        # model_loc = model_dir + 'best_model.pt'
        model_loc = os.path.join(model_dir, 'epoch-{:02d}.pt'.format(epoch_to_load))
        print("Saving/Plotting predictions of model {}".format(model_name), "at epoch {}".format(epoch_to_load))

        model_state_dict = torch.load(model_loc)['model']
        net.load_state_dict(model_state_dict)


    save_name = '{}/event_analysis_high_pt_epoch-{:02d}'.format(model_name, epoch_to_load)
    if not os.path.exists(home_dir+'results/'+model_name): os.makedirs(home_dir+'results/'+model_name)
    if process: choose_nice_events(net, test_loader, loss_cut=0.89)
    if save: df = save_class_predictions(net, test_loader, save_name, pt_cut = 0, E_cut = 0, loss_cut = 0.7)
    else: df = pd.read_csv(home_dir+'results/'+save_name+'.csv')
    print(df.head())
    print(df.describe())
    # print correlation between loss and auc
    print(df.corr())
    # print correlation between loss and auc in neutral particles
    # print(df.corr()['auc']['loss_neutral'])
    
    
    plot_multiclassification_metrics(df, save_name)

