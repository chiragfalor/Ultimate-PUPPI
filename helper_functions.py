import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

# get home directory path
with open('home_path.txt', 'r') as f:
    home_dir = f.readlines()[0].strip()

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
    