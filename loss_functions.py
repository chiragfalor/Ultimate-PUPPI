# file with all the loss functions for the model

import torch
import torch.nn as nn
import torch.nn.functional as F

def embedding_loss(pfc_enc, vtx_enc, true_vtx_id, pfc_batch, vtx_batch, neutral_weight = 1, print_bool=False):
    '''
    Args:
        pfc_enc: (particle_count, embedding_dim)
        vtx_enc: (vertex_count, embedding_dim)
        true_vtx_id: (particle_count)
        pfc_batch: (particle_count)
        vtx_batch: (vertex_count)
    Returns:
        loss: the loss from the minimizing the embedding of the particles to their nearest vertex
              + the loss from maximizing the embedding of the vertices
              + regularization loss of the embedding of vertices
    '''
    total_pfc_loss = 0
    total_vtx_loss = 0
    euclidean_loss = nn.MSELoss()
    batch_size = pfc_batch.max().item() + 1
    for i in range(batch_size):
        # get the batch index of the current batch
        pfc_indices = (pfc_batch == i)
        vtx_indices = (vtx_batch == i)
        # get the embedding of the pfc, vtx, and truth in the current batch
        pfc_enc_batch = pfc_enc[pfc_indices, :]
        vtx_enc_batch = vtx_enc[vtx_indices, :]
        truth_batch = true_vtx_id[pfc_indices].to(dtype=torch.int64)
        # take out particles which have corresponding vertices
        valid_pfc = (truth_batch >= 0)
        truth_batch = truth_batch[valid_pfc]
        pfc_enc_batch = pfc_enc_batch[valid_pfc, :]
        # the true encoding is the embedding of the true vertex
        vertex_encoding = vtx_enc_batch[truth_batch, :]
        # calculate loss between pfc encoding and vertex encoding
        pfc_loss = euclidean_loss(pfc_enc_batch, vertex_encoding)
        total_pfc_loss += pfc_loss

        if i//10 == 0:
            random_indices = torch.randperm(len(truth_batch))[:30]
            random_vtx_encoding = vertex_encoding[random_indices, :]
            for j in range(len(random_vtx_encoding)):
                for k in range(j+1, len(random_vtx_encoding)):
                    vtx_loss = -0.01*euclidean_loss(random_vtx_encoding[j, :], random_vtx_encoding[k, :])
                    total_vtx_loss += vtx_loss
        else:
            continue

    total_pfc_loss /= batch_size
    total_vtx_loss /= batch_size
    # regularize the whole embedding to keep it normalized
    reg_loss = ((torch.norm(vtx_enc, p=2, dim=1)/5)**6).mean()
    if print_bool:
        # print the losses
        print("Pfc loss: {:2f}, Vtx loss: {:2f}, Regularization loss: {:2f}".format(total_pfc_loss.item(), total_vtx_loss.item(), reg_loss.item()))
    return (total_pfc_loss + total_vtx_loss + reg_loss)/10

def contrastive_loss(pfc_embeddings, pfc_batch, true_vtx_id, num_pfc=64, c1=0.1, reg_ratio = 0.01, print_bool=False):
    '''
    Args:
        pfc_embeddings: (num_pfc, embedding_dim)
        pfc_batch: (num_pfc)
        true_vtx_id: (num_pfc)
        num_pfc: number of particles in the batch
        c1: the weight of the contrastive loss
        reg_ratio: the weight of the regularization loss
    Returns:
        loss: the loss from the minimizing the embedding of the particles to their nearest vertex
              + the loss from maximizing the embedding of the vertices
              + regularization loss of the embedding of vertices
    '''
    total_contrastive_loss = 0
    batch_size = pfc_batch.max().item() + 1
    for i in range(batch_size):
        # get the batch index of the current batch
        pfc_indices = (pfc_batch == i)
        # get the embedding of the pfc, vtx, and truth in the current batch
        pfc_enc_batch = pfc_embeddings[pfc_indices, :]
        truth_batch = true_vtx_id[pfc_indices].to(dtype=torch.int64)
        total_contrastive_loss += contrastive_loss_event(pfc_enc_batch, truth_batch, num_pfc, c1, reg_ratio, print_bool)
        print_bool = False
    total_contrastive_loss /= batch_size
    return 0.1*total_contrastive_loss
        
def contrastive_loss_event(pfc_enc, true_vtx_id, num_pfc=64, c1=1, reg_ratio = 0.01, print_bool=False):
    '''
    Calculate the contrastive loss
    input:
    pfc_enc: the encodding of the inputs
    vtx_id: the true vertex which the particle is connected to
    num_pfc: number of particles to randomly sample
    c: the ratio of positive factor for the particles of same vertex divided by the negative factor
    '''
    # loss which encourages the embedding of same particles to be close and different particles to be far
    # randomly select a set of particles to be used for contrastive loss
    random_perm = torch.randperm(len(pfc_enc))
    if len(pfc_enc) < 2*num_pfc:
        num_pfc = len(pfc_enc)//2
        # print(len(pfc_enc))
    random_indices1 = random_perm[:num_pfc]
    random_indices2 = random_perm[num_pfc:2*num_pfc]
    pfc_enc_1 = pfc_enc[random_indices1, :]
    pfc_enc_2 = pfc_enc[random_indices2, :]
    vtx_id_1 = true_vtx_id[random_indices1]
    vtx_id_2 = true_vtx_id[random_indices2]
    # number of unique vertices
    num_vtx = true_vtx_id.max().item() + 1
    # get a mask which is c if the particles are the same and -1 if they are different
    mask = -1+(c1*num_vtx+1)*(vtx_id_1 == vtx_id_2).float()
    euclidean_dist = F.pairwise_distance(pfc_enc_1, pfc_enc_2)
    loss = 10*torch.mean(mask*torch.pow(euclidean_dist, 2))
    loss += reg_ratio*((torch.norm(pfc_enc, p=2, dim=1))**4).mean()
    if print_bool:
        print("Contrastive loss: {:.2f}, Euclidean dist between particles: {:.2f}, Regularization loss: {:.2f}".format(loss.item(),torch.mean(torch.pow(euclidean_dist, 2)).item(), reg_ratio*((torch.norm(pfc_enc, p=2, dim=1))**4).mean().item()))
    return loss

def contrastive_loss_v2(pfc_enc, vtx_id, c1=0.5, c2=1, reg_ratio = 0.01, device = torch.device('cpu'), print_bool=False):
    unique_vtx = torch.unique(vtx_id)
    if len(unique_vtx) == 1:
        # if there is only one vertex, return 0 and corresponding gradient
        # print warning if there is only one vertex
        print("Warning: there is only one vertex")
        return 0
    mean_vtx = torch.zeros((len(unique_vtx), pfc_enc.shape[1])).to(device)
    for i, vtx in enumerate(unique_vtx):
        mean_vtx[i] = torch.mean(pfc_enc[vtx_id == vtx, :], dim=0)
    # get the mean of the particles of the different vertex
    mean_vtx_diff = torch.zeros((len(unique_vtx), pfc_enc.shape[1])).to(device)
    for i, vtx in enumerate(unique_vtx):
        mean_vtx_diff[i] = torch.mean(pfc_enc[vtx_id != vtx, :], dim=0)
    # get the distance between the mean of the particles of the same vertex and the mean of the particles of the different vertex
    euclidean_dist_vtx = F.pairwise_distance(mean_vtx, mean_vtx_diff)
    loss = -c2*torch.mean(torch.pow(euclidean_dist_vtx, 2))
    # add variance of the particles of the same vertex
    var_vtx = torch.zeros((len(unique_vtx), pfc_enc.shape[1])).to(device)
    for i, vtx in enumerate(unique_vtx):
        if len(pfc_enc[vtx_id == vtx, :]) > 1:
            var_vtx[i] = torch.var(pfc_enc[vtx_id == vtx, :], dim=0)
        else:
            var_vtx[i] = 0
    loss += c1*torch.mean(torch.pow(var_vtx, 2))
    loss += reg_ratio*((torch.norm(pfc_enc, p=2, dim=1))**4).mean()
    if print_bool:
        print("Contrastive loss: {}, loss from vtx distance: {}, loss from variance: {}".format(loss, -c2*torch.mean(torch.pow(euclidean_dist_vtx, 2)), c1*torch.mean(torch.pow(var_vtx, 2))))
        print("Regularization loss: {}".format(reg_ratio*((torch.norm(pfc_enc, p=2, dim=1))**4).mean()))
    if torch.isnan(loss):
        raise(ValueError)
    return loss      

def pileup_classification_loss(score, truth, neutral_mask = None, neutral_ratio = 1):
    '''
    Calculate the classification loss between primary and pileup particles
    input:
    pred: the prediction of the primary vs pileup classification
    truth: the truth of the primary vs pileup classification
    neutral_mask: mask with neutral particles
    neutral_ratio: the ratio of emphasis on neutral particles
    '''
    if neutral_mask is None or neutral_ratio == 1:
        loss = F.binary_cross_entropy_with_logits(score, truth)
    else:
        neutral_scores, neutral_truth = score[neutral_mask], truth[neutral_mask]
        charged_scores, charged_truth = score[~neutral_mask], truth[~neutral_mask]
        loss = (F.binary_cross_entropy_with_logits(charged_scores, charged_truth) + neutral_ratio*F.binary_cross_entropy_with_logits(neutral_scores, neutral_truth))/(1+neutral_ratio)
    return loss

def multiclassification_loss(score, truth, neutral_mask = None, neutral_ratio = 1, weighting = None, pt = None):
    '''
    Calculate classification loss for multiple vertex prediction
    input:
    score : (number of particles, number of classes), score of each particle being in each class
    truth : (number of particles), the true class of each particle (0-indexed)
    neutral_mask : (number of particles), mask
    neutral_ratio : (float), the ratio of emphasis on neutral particles
    '''
    vtx_classes = score.shape[1]
    # weight the loss function based on inversely number of particles in each class
    def get_num_weights(truth):
        try:
            weights = torch.zeros(vtx_classes).to(truth.device)
            for i in range(vtx_classes):
                weights[i] = 1/torch.sum(truth == i).item()
            weights = weights/torch.sum(weights)
            return weights
        except:
            print("Error in weighting loss function")
            return None
    def cross_entropy_loss(score, truth, pt=None):
        if weighting is None:
            return F.cross_entropy(score, truth)
        elif weighting == 'num' or pt is None:
            return F.cross_entropy(score, truth, weight=get_num_weights(truth))
        elif weighting == 'pt':
            losses = F.cross_entropy(score, truth, reduction='none')
            loss = torch.sum(losses*pt)/torch.sum(pt)
            return loss



    if neutral_mask is None or neutral_ratio == 1:
        loss = cross_entropy_loss(score, truth, pt)
    else:
        neutral_scores, neutral_truth = score[neutral_mask], truth[neutral_mask]
        charged_scores, charged_truth = score[~neutral_mask], truth[~neutral_mask]
        loss = (cross_entropy_loss(charged_scores, charged_truth, pt[~neutral_mask]) + neutral_ratio*cross_entropy_loss(neutral_scores, neutral_truth, pt[neutral_mask]))/(1+neutral_ratio)
    return loss

def MET_loss_fn(x_pfc, x_pfc_batch_mask, scores):
    '''
    enforces momentum conservation in the vertices
    input:
    x_pfc: (number of particles, number of features), the features of the particles
    x_pfc_batch: (number of particles, number of features), the features of the particles in the batch
    scores: (number of particles, number of classes), the score of each particle being in each class
    '''
    number_of_particles = x_pfc.shape[0]
    vtx_classes = scores.shape[1]
    prob = torch.softmax(scores, dim=1)
    MET_loss = 0
    batch_size = x_pfc_batch_mask.max().item()+1
    for i in range(batch_size):
        x_pfc_batch = x_pfc[x_pfc_batch_mask == i]
        prob_batch = prob[x_pfc_batch_mask == i]
        for j in range(vtx_classes):
            total_px = torch.sum(x_pfc_batch[:, 0]*prob_batch[:, j])
            total_py = torch.sum(x_pfc_batch[:, 1]*prob_batch[:, j])
            MET_loss += total_px**2+total_py**2
    return 32*MET_loss/number_of_particles
    
        

def combined_loss_fn(data, z_pred, pfc_embeddings = None, vtx_embeddings = None, embedding_loss_weight=1, neutral_weight = 1, print_bool=False):
    '''
    Computes the combined loss including regression loss and embedding loss
    '''
    if embedding_loss_weight > 0:
        if vtx_embeddings is None:
            # use contrastive loss if no vertex embedding is provided
            emb_loss = contrastive_loss(pfc_embeddings, data.truth.int(), print_bool=print_bool)
        else:
            # use embedding loss
            emb_loss = embedding_loss(pfc_embeddings, vtx_embeddings, data.truth.int(), pfc_batch = data.x_pfc_batch, vtx_batch = data.x_vtx_batch, print_bool=print_bool)
    else:
        emb_loss = 0
    # calculate the regression loss
    # regression_loss_fn = nn.MSELoss()
    regression_loss_fn = nn.L1Loss()
    z_pred = z_pred.squeeze()
    if neutral_weight != 1:
        neutral_idx, charged_idx = torch.nonzero(data.x_pfc[:, -2] == 0).squeeze(), torch.nonzero(data.x_pfc[:, -2] != 0).squeeze()
        neutral_z_pred, charged_z_pred = z_pred[neutral_idx], z_pred[charged_idx]
        neutral_z_true, charged_z_true = data.y[neutral_idx], data.y[charged_idx]
        neutral_loss = regression_loss_fn(neutral_z_pred, neutral_z_true)
        charged_loss = regression_loss_fn(charged_z_pred, charged_z_true)
        regression_loss = (neutral_weight*neutral_loss + charged_loss)/(neutral_weight + 1)
    else:
        regression_loss = regression_loss_fn(z_pred, data.y)
    loss = embedding_loss_weight*emb_loss + regression_loss
    if print_bool:
        print("Total loss: {:.2f}, Embedding loss: {:.2f}, Regression loss: {:.2f}".format(loss, embedding_loss_weight*emb_loss, regression_loss))
    return loss


def combined_classification_embedding_loss_puppi(data, score, pfc_embeddings = None, vtx_embeddings = None, embedding_loss_weight=0.01, neutral_weight = 1, MET_loss_weight = 0, vtx_classes = 1, classification_weighting = 'pt', print_bool=False):
    '''
    Computes the combined loss including classification loss and embedding loss
    '''
    if embedding_loss_weight > 0:
        if vtx_embeddings is None:
            # use contrastive loss if no vertex embedding is provided
            emb_loss = contrastive_loss(pfc_embeddings, data.x_pfc_batch, data.truth.int(), print_bool=print_bool)
        else:
            # use embedding loss
            emb_loss = embedding_loss(pfc_embeddings, vtx_embeddings, data.truth.int(), pfc_batch = data.x_pfc_batch, vtx_batch = data.x_vtx_batch, print_bool=print_bool)
    else:
        emb_loss = 0
    # calculate the classification loss
    
    neutral_mask = data.x_pfc[:, -2] == 0
    pt = torch.sqrt(data.x_pfc[:, 0]**2 + data.x_pfc[:, 1]**2).float()
    # pt = pt[neutral_mask]
    # truth = (data.truth != 0).long()   # process truth here
    truth = data.truth.long()
    # clip the truth to the number of classes
    truth = truth.clamp(max = vtx_classes)
    # replace truth -1 to truth = vtx_classes
    truth[truth == -1] = vtx_classes
    classification_loss = 100*multiclassification_loss(score, truth, neutral_mask, neutral_weight, weighting=classification_weighting, pt=pt)

    # calculate the MET loss
    MET_loss = MET_loss_fn(data.x_pfc, data.x_pfc_batch, score)

    # classification_loss = 100*pileup_classification_loss(score, truth, neutral_mask, neutral_weight)
    loss = embedding_loss_weight*emb_loss + classification_loss + MET_loss_weight*MET_loss
    if print_bool:
        print("Total loss: {:.2f}, Embedding loss: {:.2f}, Classification loss: {:.2f}, MET loss: {:.2f}".format(loss, embedding_loss_weight*emb_loss, classification_loss, MET_loss_weight*MET_loss))
    return loss

