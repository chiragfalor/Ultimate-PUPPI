# file with all the loss functions for the model

import torch
import torch.nn as nn

def embedding_loss(pfc_enc, vtx_enc, true_vtx_id, pfc_batch, vtx_batch, print_loss=False):
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
    reg_loss = ((torch.norm(vtx_enc, p=2, dim=1))**6).mean()
    if print_loss:
        # print the losses
        print("Pfc loss: ", total_pfc_loss.item(), " Vtx loss: ", total_vtx_loss.item(), " Reg loss: ", reg_loss.item())
    return total_pfc_loss + total_vtx_loss + reg_loss

