from operator import add
from helper_functions import *
from dataset_graph_loader import UPuppiV0
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, remove_self_loops)
import torch_geometric
print(torch_geometric.__version__)
torch.manual_seed(0)

data_test = UPuppiV0(home_dir + "test5/")
data = data_test[0]

print(data)
data = train_test_split_edges(data, 0.05, 0.1)
print(data)
print(add_self_loops(data.train_pos_edge_index))
edge_index, _ = add_self_loops(data.train_pos_edge_index)
data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
print(data.train_neg_edge_index)
print(data.edge_weight)