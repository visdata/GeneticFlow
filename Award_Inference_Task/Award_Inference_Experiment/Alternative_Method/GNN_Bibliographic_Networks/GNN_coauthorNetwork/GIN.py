import torch.nn.functional as F
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from ogb.graphproppred import PygGraphPropPredDataset as Dataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv,GINEConv, global_mean_pool

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=2, num_layers=1,
                 dropout=0.5):
        super().__init__()

        self.dropout = dropout

        
        self.atom_encoder = AtomEncoder(hidden_channels)
        self.bond_encoder = BondEncoder(hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            if (_==0):
                nn = Sequential(
                Linear(6, 2 * hidden_channels),
                BatchNorm(2 * hidden_channels),
                ReLU(),
                Linear(2 * hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
            )
            else:
                nn = Sequential(
                    Linear(hidden_channels, 2 * hidden_channels),
                    BatchNorm(2 * hidden_channels),
                    ReLU(),
                    Linear(2 * hidden_channels, hidden_channels),
                    BatchNorm(hidden_channels),
                    ReLU(),
                )
            self.convs.append(GINEConv(nn, train_eps=True, edge_dim=1))

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data, paper_count):
        # print(data)
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        
        
        x = global_mean_pool(x, batch)
        # paper_count = torch.unsqueeze(paper_count,1)
        # x = torch.cat((x, paper_count),1)
        
        x = F.log_softmax(self.lin(x))
        return x