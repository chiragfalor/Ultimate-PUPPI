import os, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch import nn
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader


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
    elif model == "modelv2" or model == "modelv2_neg" or model == "modelv2_nz0" or model == "modelv2_nz199" or model == "modelv2_orig" or model=="modelv2_contrastive" or model=="modelv2_newdata" or model == "modelv2_analysis":
        from models.modelv2 import Net
    elif model == "modelv3":
        from models.modelv3 import Net
    elif model == "Dynamic_GATv2":
        from models.Dynamic_GATv2 import Net
    elif model == "DynamicTransformer":
        from models.DynamicTransformer import Net
    elif model == "DynamicPointTransformer":
        from models.DynamicPointTransformer import Net
    elif model == "embedding_GCN" or model == "embedding_GCN_allvtx" or model == "embedding_GCN_nocheating":
        from models.embedding_GCN import Net
    else:
        raise(Exception("Model not found"))
    return Net


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


def plot_2_embeddings(embeddings1, embeddings2, save_name, color1=None, color2=None, label1=None, label2=None, colored=False):
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
        plt.scatter(embeddings2_2d[:, 0], embeddings2_2d[:, 1], c=color2, cmap=cm.get_cmap('jet'), marker='*', s=100)
        cbar.set_label('z value')
        
    else:
        plt.scatter(embeddings1_2d[:, 0], embeddings1_2d[:, 1], c='red', marker='.', s=10)
        plt.scatter(embeddings2_2d[:, 0], embeddings2_2d[:, 1], c='blue', marker='.', s=0.5)
    # add legend
    if label1 is not None:
        plt.legend([label1, label2])
    plt.savefig(home_dir + 'results/{}'.format(save_name), bbox_inches='tight')
    plt.close()
    

def pngs_to_gif(png_dir, gif_name, size=(500, 500), fps=5):
    '''
    png_dir: string
    gif_name: string
    return: None
    '''
    # get all pngs in the directory
    png_list = os.listdir(png_dir)
    # check the extension of each png
    png_list = [png for png in png_list if png.endswith('.png')]
    # sort the pngs by name
    png_list.sort()
    # create a list of images
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
    # save the gif in the same directory as the pngs
    save_loc = png_dir + gif_name + '.gif'
    # save the gif
    images[0].save(save_loc, save_all=True, append_images=images[1:], duration=1000/fps, loop=0)


def save_predictions(net, data_loader, save_name):
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
    # add neutral loss and total loss
    plt.text(70, -200, "Total loss: {:.2f}".format(total_loss))
    plt.text(50, -220, "Neutral loss: {:.2f}".format(neutral_loss))
    # add legend
    plt.legend(['Charged', 'Neutral'])
    # add axis labels
    plt.xlabel('True z')
    plt.ylabel('Predicted z')
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
        save_name = '/'+ model_name + '/'+ model_name + '_' + epoch[:-3]
        df = save_predictions(net, data_loader, save_name)
        plot_z_pred_z_true(df, save_name)

    pngs_to_gif(save_dir, model_name + '_evolution')


