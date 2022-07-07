import os, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch import nn
from sklearn.decomposition import PCA
from scipy import stats
from PIL import Image
from tqdm import tqdm
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
from sklearn import metrics


# get home directory path
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()


def get_neural_net(model_name):
    '''
    model_name: string
    return: class object of the neural network
    '''
    model = model_name
    
    if model == "DynamicGCN":
        from models.DynamicGCN import Net
    elif model == "GAT":
        from models.GAT import Net
    elif model == "GravNetConv":
        from models.GravNetConv import Net
    elif model == "No_Encode_grav_net":
        from models.No_Encode_grav_net import Net
    elif model == "combined_model2" or model == "combined_model":
        from models.model import Net
    elif model[:7] == "modelv2":   # or model == "modelv2_neg" or model == "modelv2_nz0" or model == "modelv2_nz199" or model == "modelv2_orig" or model=="modelv2_contrastive" or model=="modelv2_newdata" or model == "modelv2_analysis" or model == "modelv2_random_z":
        from models.modelv2 import Net
    elif model[:7] == "modelv3":
        from models.modelv3 import Net
    elif model == "Dynamic_GATv2":
        from models.Dynamic_GATv2 import Net
    elif model == "DynamicTransformer":
        from models.DynamicTransformer import Net
    elif model == "DynamicPointTransformer":
        from models.DynamicPointTransformer import Net
    elif model == "embedding_GCN" or model == "embedding_GCN_allvtx" or model == "embedding_GCN_nocheating":
        from models.embedding_GCN import Net
    elif model[:14] == "vtx_pred_model":
        from models.emb_v2 import Net
    elif model[:6] == "pileup":
        from models.classification_model_puppi import Net
    elif model[:15] == "multiclassifier":
        from models.multiclass_model import Net
    else:
        raise(Exception("Model not found"))
    return Net


def process_data(data):
    '''
    Apply data processing as needed and return the processed data.
    '''
    return data
    data.truth = (data.truth != 0).int()
    neutral_idx = torch.nonzero(data.x_pfc[:,-2] == 0).squeeze()
    # randomly select half of the neutral particles
    half_neutral_idx = neutral_idx[torch.randperm(neutral_idx.shape[0])[:int(neutral_idx.shape[0]/2)]]
    # multiply the neutral particles by -1
    data.x_pfc[half_neutral_idx, -1] *= -1
    # get the indices where truth is not 0
    pileup_idx = torch.nonzero(data.truth != 0).squeeze()
    # keep only the pileup particles
    data.x_pfc = data.x_pfc[pileup_idx]
    data.x_pfc_batch = data.x_pfc_batch[pileup_idx]
    data.truth = data.truth[pileup_idx]
    data.y = data.y[pileup_idx]
    # data.x_vtx = data.x_vtx[1:, :]
    # data.x_vtx_batch = data.x_vtx_batch[1:]
    return data

def process_truth(truth, vtx_classes):
    '''
    truth: (N)
    vtx_classes: int
    return: (N)
    clamps the truth to the number of vertex classes + pileup vertex
    '''
    # clamp the truth to vtx_classes
    truth = torch.clamp(truth, max=vtx_classes)
    # for truth -1 (pileup), make it vtx_classes
    truth[truth == -1] = vtx_classes
    return truth

def vertex_predictor(particle_embedding, pfc_embeddings, true_vertex, k=1):
    '''
    particle_embedding: (1, embedding_dim)
    pfc_embeddings: (N, embedding_dim)
    true_vertex: (N)
    k: int
    return: (N)
    returns the index of the most common vertex among the k nearest neighbors of particle_embedding
    '''
    # get the distance between particle_embedding and pfc_embeddings
    distance = torch.norm(pfc_embeddings - particle_embedding, p=2, dim=1)
    # get the indices of the k nearest neighbors
    indices = torch.topk(distance, k, largest=False)[1]
    # get the labels of the k nearest neighbors
    labels = true_vertex[indices]
    # get the majority vote of the labels
    majority_vote = torch.mode(labels)[0]
    return majority_vote


def plot_2_embeddings(embeddings1, embeddings2, save_name, color1=None, color2=None, label1=None, label2=None, colored=False, colorbar_label = 'z true'):
    '''
    embeddings1: (n1, embedding_dim)
    embeddings2: (n2, embedding_dim)
    labels1: (n1)
    labels2: (n2)
    save_path: string
    colored: bool
    return: None
    Saves a PCA plot of:
    if colored, embeddings1 represented by dots and embeddings2 represented by stars colored by their labels
    if not colored, embeddings1 represented by red dots and embeddings2 represented by blue small dots
    '''
    pca = PCA(n_components=2)
    embeddings = np.concatenate((embeddings1, embeddings2), axis=0)
    embeddings_2d = pca.fit_transform(embeddings)
    embeddings1_2d = embeddings_2d[:embeddings1.shape[0]]
    embeddings2_2d = embeddings_2d[embeddings1.shape[0]:]
    if colored:
        plt.scatter(embeddings1_2d[:, 0], embeddings1_2d[:, 1], c=color1, cmap=cm.get_cmap('jet'))
        cbar = plt.colorbar()
        plt.scatter(embeddings2_2d[:, 0], embeddings2_2d[:, 1], c=color2, cmap=cm.get_cmap('jet'), marker='*', s=50)
        cbar.set_label(colorbar_label)
        
    else:
        plt.scatter(embeddings1_2d[:, 0], embeddings1_2d[:, 1], c='red', marker='.', s=10)
        plt.scatter(embeddings2_2d[:, 0], embeddings2_2d[:, 1], c='blue', marker='.', s=0.5)
    # add legend
    if label1 is not None:
        plt.legend([label1, label2])
    plt.savefig(home_dir + 'results/{}'.format(save_name), bbox_inches='tight')
    plt.close()
    

def pngs_to_gif(png_dir, gif_name, size=(580, 450), fps=5):
    '''
    png_dir: string
    gif_name: string
    return: None
    '''
    png_list = os.listdir(png_dir)
    png_list = [png for png in png_list if png.endswith('.png')]
    # sort the pngs by number at the end of the file name
    # png_list.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    png_list.sort()   # sort by file name
    images = []

    for png in png_list:
        # get the path of the png
        png_path = os.path.join(png_dir, png)
        # read the png
        image = Image.open(png_path)
        # resize the image
        image = image.resize(size)
        # add the image to the list
        images.append(image)
    
    save_loc = png_dir + gif_name + '.gif'
    images[0].save(save_loc, save_all=True, append_images=images[1:], duration=1000/fps, loop=0)


def save_z_predictions(net, data_loader, save_name):
    '''
    Saves the data and predictions of the network for all the data in the data_loader
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    total_loss = 0
    # initialize np arrays to save the data and predictions
    z_pred = None
    for data in tqdm(data_loader):
        data = process_data(data)
        data = data.to(device)
        with torch.no_grad():
            z_out = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)[0]
            loss = nn.MSELoss()(z_out.squeeze(), data.y)
            total_loss += loss.item()
            if z_pred is None:
                z_pred = torch.squeeze(z_out).detach().cpu().numpy()
                z_true = data.y.detach().cpu().numpy()
                vtx_truth = data.truth.detach().cpu().numpy() 
                charge = data.x_pfc[:, -2].detach().cpu().numpy()
            else:
                z_pred = np.concatenate((z_pred, torch.squeeze(z_out).detach().cpu().numpy()), axis=0)
                z_true = np.concatenate((z_true, data.y.detach().cpu().numpy()), axis=0)
                vtx_truth = np.concatenate((vtx_truth, data.truth.detach().cpu().numpy()), axis=0)
                charge = np.concatenate((charge, data.x_pfc[:, -2].detach().cpu().numpy()), axis=0)
    print("Total loss: {}".format(total_loss/len(data_loader)))
    data_dict = {'z_pred': z_pred, 'z_true': z_true, 'charge': charge, 'vtx_truth': vtx_truth}
    df = pd.DataFrame(data_dict)
    df.to_csv(home_dir + 'results/{}.csv'.format(save_name), index=False)
    print("Saved data and predictions to results/{}.csv".format(save_name))
    return df


def save_class_predictions(net, data_loader, save_name):
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
        data = data.to(device)
        with torch.no_grad():
            scores_batch = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)[0]
            vtx_classes = scores_batch.shape[1] - 1
            class_prob_batch = nn.Softmax(dim=1)(scores_batch)
            class_true_batch = process_truth(data.truth, vtx_classes).long()
            pred_batch = torch.argmax(class_prob_batch, dim=1)
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
    print("Total loss: {}".format(total_loss), "with {} vertex classes".format(vtx_classes))
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


def plot_z_pred_z_true(df, save_name):
    '''
    df: pandas dataframe
    save_name: string
    return: None
    '''
    # calculate total test loss
    total_loss = nn.MSELoss()(torch.tensor(df.z_pred.values), torch.tensor(df.z_true.values)).item()
    # separate charged and neutral particles
    charged = df[df.charge != 0]
    neutral = df[df.charge == 0]
    # calculate neutral loss
    neutral_loss = nn.MSELoss()(torch.tensor(neutral.z_pred.values), torch.tensor(neutral.z_true.values)).item()
    # plot charged particles with red dots and neutral particles with blue dots
    plt.scatter(charged.z_true, charged.z_pred, c='red', marker='.', s=1)
    plt.scatter(neutral.z_true, neutral.z_pred, c='blue', marker='.', s=1)
    # plot two regression for charged and neutral particles with slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(neutral.z_true, neutral.z_pred)
    # plt.plot(neutral.z_true, slope*neutral.z_true + intercept, c='blue', label=slope)
    # Beautify the plot
    plt.title(save_name.split('/')[-1])
    plt.text(70, -200, "Total loss: {:.2f}".format(total_loss))
    plt.text(50, -220, "Neutral loss: {:.2f}".format(neutral_loss))
    # slopes
    # plt.text(-20, 200, "Neutral slope: {:.2f}".format(slope))
    plt.legend(['Charged', 'Neutral'], loc='upper left')
    plt.xlabel('True z')
    plt.ylabel('Predicted z')
    plt.xlim(-250, 250)
    plt.ylim(-250, 250)
    plt.savefig(home_dir + 'results/{}.png'.format(save_name), bbox_inches='tight')
    plt.close()


def make_model_evolution_gif(net, model_name, data_loader):
    model_dir = home_dir + 'models/{}/'.format(model_name)
    if not os.path.exists(model_dir):
        raise(Exception("Model directory {} does not exist".format(model_dir)))
    
    save_dir = home_dir + 'results/{}/'.format(model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    epoch_list = os.listdir(model_dir)
    epoch_list = [epoch for epoch in epoch_list if epoch.endswith('.pt')]
    epoch_list.sort()

    for epoch in epoch_list:
        net.load_state_dict(torch.load(model_dir + epoch)['model'])
        save_name = model_name + '/'+ model_name + '_' + epoch[:-3]
        df = save_z_predictions(net, data_loader, save_name)
        # df = pd.read_csv(home_dir + 'results/{}.csv'.format(save_name))
        plot_z_pred_z_true(df, save_name)

    pngs_to_gif(save_dir, model_name + '_evolution')


def plot_z_predictions3(df, model_name):
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

    plt.savefig(home_dir + 'results/{}_zpred_vs_ztrue.png'.format(model_name), bbox_inches='tight')
    plt.close()

def plot_class_predictions2(df, save_name):
    # make 2 histograms:
    # 1. class prediction vs. class true (for all particles)
    # 2. class prediction vs. class true (for neutral particles)
    # on the same figure
    fig, axs = plt.subplots(2, 1, figsize=(10,12))
    # make a red histogram of primary particles and a blue histogram of pileup particles
    axs[0].hist(1-df[df['class_true'] == 1]['class_prob_0'], bins=np.arange(0,1,0.01), color='blue', label='pileup')
    axs[0].hist(1-df[df['class_true'] == 0]['class_prob_0'], bins=np.arange(0,1,0.01), color='red', label='primary')
    axs[0].set_xlabel('class_pred')
    axs[0].set_ylabel('count')
    axs[0].set_title('class_pred vs class_true for all particles')
    axs[0].legend()
    # for neutral particles
    df = df[df['charge'] == 0]
    axs[1].hist(1-df[df['class_true'] == 1]['class_prob_0'], bins=np.arange(0,1,0.01), color='blue', label='pileup')
    axs[1].hist(1-df[df['class_true'] == 0]['class_prob_0'], bins=np.arange(0,1,0.01), color='red', label='primary')
    axs[1].set_xlabel('class_pred')
    axs[1].set_ylabel('count')
    axs[1].set_title('class_pred vs class_true for neutral particles')
    axs[1].legend()
    # fig.suptitle(save_name.split('/')[-1])
    plt.savefig(home_dir + 'results/{}.png'.format(save_name), bbox_inches='tight')
    plt.close()


def plot_binary_roc_auc_score(df, save_name):
    pred = 1-df.class_prob_0.values
    true = df.class_true.values
    # metrics.RocCurveDisplay.from_predictions(true, pred).plot()
    fpr, tpr, thresholds = metrics.roc_curve(true, pred)
    roc_auc = metrics.auc(fpr, tpr)
    # plot log linear ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # set y scale to logit
    # plt.yscale('logit')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # add title and save
    plt.title(save_name.split('/')[-1] + '\nAUC score: {:.2f}'.format(roc_auc))
    plt.savefig(home_dir + 'results/{}.png'.format(save_name), bbox_inches='tight')
    plt.close()
