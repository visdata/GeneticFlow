import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, GlobalAttention, GraphMultisetTransformer, Set2Set, GlobalAttention
from torch_geometric.nn import ResGatedGraphConv,ChebConv,SAGEConv,GCNConv,GATConv,TransformerConv,AGNNConv,EdgePooling,GraphConv,GCN2Conv,TopKPooling,SAGPooling
from torch_geometric.nn import GINConv,GATv2Conv,ASAPooling,LEConv,MFConv,SGConv,ARMAConv,TAGConv
from torch_geometric.utils import get_laplacian
from layers import GCN, HGPSLPool
from torch.nn import Linear, ReLU, Sequential
from torch.nn import BatchNorm1d as BatchNorm
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch_geometric.transforms as T


import matplotlib.pyplot as plt

def get_conv(conv_name):
    if conv_name == 'GCNConv':
        return GCNConv
    elif conv_name == 'ChebConv':
        return ChebConv
    elif conv_name == 'SAGEConv':
        return SAGEConv
    elif conv_name == 'GraphConv':
        return GraphConv
    elif conv_name == 'GATConv':
        return GATConv
    elif conv_name == 'TAGConv':
        return TAGConv
    elif conv_name == 'ARMAConv':
        return ARMAConv
    elif conv_name == 'SGConv':
        return SGConv
    elif conv_name == 'MFConv':
        return MFConv
    elif conv_name == 'LEconv':
        return LEConv
    elif conv_name == 'GINConv':
        return GINConv

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, enhance=False):
        super(MLP, self).__init__()
        self.enhance = enhance
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        if enhance:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(0.5) 
        # self.para_init()

    def para_init(self):
        for p in self.parameters():
            nn.init.constant_(p,1)

    def forward(self, x):
        x = self.fc1(x.to('cuda:0'))
        if self.enhance:
            x = self.bn1(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.enhance:
            x = self.bn2(x)
        x = torch.relu(x)
        if self.enhance:
            x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        # x = F.log_softmax(self.fc3(x), dim=-1)
        return x


class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.sample = args.sample_neighbor
        self.sparse = args.sparse_attention
        self.sl = args.structure_learning
        self.lamb = args.lamb
        self.edgeconv = MLP(10, 64, 1).to('cuda:0')
        self.conv_name = args.conv_name
        self.pool_name = args.pool_name
        # self.feature_mlp = MLP(6, 32, 32).to('cuda:0')
        Conv = get_conv(self.conv_name)

        # GCN2
        # alpha=0.1
        # theta=0.5
        # shared_weights=False
        # self.lins = torch.nn.ModuleList()
        # self.lins.append(Linear(self.num_features, self.nhid))
        # self.lins.append(Linear(self.nhid, self.nhid))
        # self.convs = torch.nn.ModuleList()
        # for layer in range(64):
        #     self.convs.append(
        #         GCN2Conv(self.nhid, alpha, theta, layer + 1,
        #                  shared_weights, normalize=False))

        if self.conv_name == 'ChebConv':
            self.conv1 = ChebConv(self.num_features, self.nhid, 1)
            self.conv2 = ChebConv(self.nhid, self.nhid, 1)
            self.conv3 = ChebConv(self.nhid, self.nhid, 1)
        elif self.conv_name == 'GINConv':
            hidden_channels=self.nhid
            nn1 = Sequential(
                Linear(6, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
            )
            self.conv1=GINConv(nn1, train_eps=True)
            nn2 = Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
            )
            self.conv2=GINConv(nn2, train_eps=True)
            nn3 = Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                BatchNorm(hidden_channels),
                ReLU(),
            )
            self.conv3=GINConv(nn3, train_eps=True)
        else:
            self.conv1 = Conv(self.num_features, self.nhid)
            self.conv2 = Conv(self.nhid, self.nhid)
            self.conv3 = Conv(self.nhid, self.nhid)


        if self.pool_name == 'SAGPooling':
            self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio, GNN=GCNConv)
            self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio, GNN=GCNConv)
        elif self.pool_name == 'TopKPooling':
            self.pool1 = TopKPooling(self.nhid, ratio=self.pooling_ratio)
            self.pool2 = TopKPooling(self.nhid, ratio=self.pooling_ratio)
        elif self.pool_name == 'ASAPooling':
            self.pool1 = ASAPooling(self.nhid, ratio=self.pooling_ratio, dropout=0.5, GNN=GCNConv,
                                add_self_loops=False)
            self.pool2 = ASAPooling(self.nhid, ratio=self.pooling_ratio, dropout=0.5, GNN=GCNConv,
                                add_self_loops=False)
        elif self.pool_name == 'EdgePooling':
            self.pool1 = EdgePooling(self.nhid, edge_score_method=EdgePooling.compute_edge_score_softmax, dropout=0.2, add_to_edge_score=1)
            self.pool2 = EdgePooling(self.nhid, edge_score_method=EdgePooling.compute_edge_score_softmax, dropout=0.2, add_to_edge_score=1)
        elif self.pool_name == 'HGPSLPool':
            self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
            self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        self.lin_combine = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin1 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin2 = torch.nn.Linear(self.nhid // 2 + 1, self.nhid // 4)
        self.lin3 = torch.nn.Linear(self.nhid // 4, self.num_classes)

    def forward(self, data, paper_count):

        # gdc
        # gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
        #         normalization_out='col',
        #         diffusion_kwargs=dict(method='ppr', alpha=0.05),
        #         sparsification_kwargs=dict(method='topk', k=128,
        #                                    dim=0), exact=True)
        # data = gdc(data)

        x, edge_index, batch = data.x.to('cuda:0'), data.edge_index.to('cuda:0'), data.batch.to('cuda:0')

        # x_reverse, edge_index_reverse, batch_reverse = x, edge_index, batch
        # idx = torch.LongTensor([1,0]).to('cuda:0')
        # edge_index_reverse=edge_index_reverse.index_select(0, idx)
        
        # draw graph
        # x_cpu=x.cpu().detach().numpy()
        # edge_index_cpu=edge_index.cpu().detach().numpy()
        # batch_cpu=batch.cpu().detach().numpy()
        # X=[[] for i in range(batch_cpu[batch_cpu.shape[0]-1]+1)]
        # Edge_index=[[[],[]] for i in range(batch_cpu[batch_cpu.shape[0]-1]+1)]
        # count=0
        # for i in range(x_cpu.shape[0]):
        #     X[batch_cpu[i]].append(x_cpu[i])
        # for i in range(edge_index_cpu.shape[1]):
        #     Edge_index[batch_cpu[edge_index_cpu[0][i]]][0].append(edge_index_cpu[0][i])
        #     Edge_index[batch_cpu[edge_index_cpu[0][i]]][1].append(edge_index_cpu[1][i])
        # Edge_index[1]=np.array(Edge_index[1])
        # idx = np.array(np.unique(Edge_index[1].reshape(-1)), dtype=np.dtype(int))
        # idx_map = {j: i for i, j in enumerate(idx)}
        # Edge_index[1]=np.array(list(map(idx_map.get, Edge_index[1].flatten())),
        #                             dtype=np.int32).reshape(Edge_index[1].shape)
        # G = Data(x=torch.from_numpy(np.array(X[1])), edge_index=torch.from_numpy(np.array(Edge_index[1])))
        # print(G.num_nodes)
        # print(np.array(X[1]).shape)
        # G = to_networkx(G)
        # print(G.nodes(),Edge_index[1])
        # nx.draw(G,node_size=20)
        # plt.savefig("origin.png")
        # plt.clf()    
        
        graph_ave_feature = gap(x, batch)
        edge_attr = None
        # edge_attr = data.edge_attr
        # edge_attr = self.edgeconv(data.edge_attr)

        # compute proba
        # proba1=torch.flatten(edge_attr).to('cuda:0')
        # proba2=torch.flatten(edge_proba).to('cuda:0')
        # proba_distance=proba1-proba2
        # print(torch.mean(proba1),torch.mean(proba2))
        # norm = torch.norm(proba_distance, p=1, dim=0)
        # norm = norm/proba1.shape[0]
        if self.conv_name == 'ChebConv':
            x = F.relu(self.conv1(x.float(), edge_index, edge_attr, batch))
        else:
            x = F.relu(self.conv1(x.float(), edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # if self.pool_name == 'SAGPooling':
        #     x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        # elif self.pool_name == 'TopKPooling':
        #     x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        # elif self.pool_name == 'ASAPooling':
        #     x, edge_index, edge_attr, batch, _ = self.pool1(x, edge_index, edge_attr, batch)
        # elif self.pool_name == 'EdgePooling':
        #     x, edge_index, batch, _ = self.pool1(x, edge_index, batch)
        # elif self.pool_name == 'HGPSLPool':
        #     x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
        x1 = gap(x, batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        edge_attr=None
        if self.conv_name == 'ChebConv':
            x = F.relu(self.conv2(x, edge_index, edge_attr, batch))
        else:
            x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # if self.pool_name == 'SAGPooling':
        #     x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        # elif self.pool_name == 'TopKPooling':
        #     x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        # elif self.pool_name == 'ASAPooling':
        #     x, edge_index, edge_attr, batch, _ = self.pool2(x, edge_index, edge_attr, batch)
        # elif self.pool_name == 'EdgePooling':
        #     x, edge_index, batch, _ = self.pool2(x, edge_index, batch)
        # elif self.pool_name == 'HGPSLPool':
        #     x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = gap(x, batch)

        edge_attr=None
        if self.conv_name == 'ChebConv':
            x = F.relu(self.conv3(x, edge_index, edge_attr, batch))
        else:
            x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x3 = gap(x, batch)
        
        # x = F.relu(x1)
        x = F.relu(x1)+F.relu(x2)+F.relu(x3)

        # edge_attr=None
        # if self.conv_name == 'ChebConv':
        #     x_reverse = F.relu(self.conv1(x_reverse.float(), edge_index_reverse, edge_attr, batch_reverse))
        # else:
        #     x_reverse = F.relu(self.conv1(x_reverse.float(), edge_index_reverse, edge_attr))
        # x_reverse = F.dropout(x_reverse, p=self.dropout_ratio, training=self.training)
        # x1_reverse = gap(x_reverse, batch_reverse)
        # edge_attr=None
        # if self.conv_name == 'ChebConv':
        #     x_reverse = F.relu(self.conv2(x_reverse, edge_index_reverse, edge_attr, batch_reverse))
        # else:
        #     x_reverse = F.relu(self.conv2(x_reverse, edge_index_reverse, edge_attr))
        # x_reverse = F.dropout(x_reverse, p=self.dropout_ratio, training=self.training)
        # x2_reverse = gap(x_reverse, batch_reverse)
        # edge_attr=None
        # if self.conv_name == 'ChebConv':
        #     x_reverse = F.relu(self.conv3(x_reverse, edge_index_reverse, edge_attr, batch_reverse))
        # else:
        #     x_reverse = F.relu(self.conv3(x_reverse, edge_index_reverse, edge_attr))
        # x_reverse = F.dropout(x_reverse, p=self.dropout_ratio, training=self.training)
        # x3_reverse = gap(x_reverse, batch_reverse)

        # x_reverse = F.relu(x1_reverse)+F.relu(x2_reverse)+F.relu(x3_reverse)
        # x = F.relu(self.lin_combine(torch.cat((x, x_reverse),1)))

        # GCN2 forword
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = x_0 = self.lins[0](x).relu()
        # for conv in self.convs:
        #     x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        #     x = conv(x, x_0, edge_index, None)
        #     x = x.relu()
        # x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = self.lins[1](x)
        # x = gap(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)   
        paper_count = torch.unsqueeze(paper_count,1)    
        x = torch.cat((x, paper_count),1)
        x = F.relu(self.lin2(x))    
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


# MEWISPool
class MEWISPool(nn.Module):
    def __init__(self, hidden_dim):
        super(MEWISPool, self).__init__()

        self.gc1 = GINConv(MLP(1, hidden_dim, hidden_dim))
        self.gc2 = GINConv(MLP(hidden_dim, hidden_dim, hidden_dim))
        self.gc3 = GINConv(MLP(hidden_dim, hidden_dim, 1))

    def forward(self, x, edge_index, batch):
        # computing the graph laplacian and adjacency matrix
        batch_nodes = batch.size(0)
        if edge_index.size(1) != 0:
            L_indices, L_values = get_laplacian(edge_index)
            L = torch.sparse.FloatTensor(L_indices, L_values, torch.Size([batch_nodes, batch_nodes]))
            A = torch.diag(torch.diag(L.to_dense())) - L.to_dense()

            # entropy computation
            entropies = self.compute_entropy(x, L, A, batch)  # Eq. (8)
        else:
            A = torch.zeros([batch_nodes, batch_nodes]).to('cuda:0')
            norm = torch.norm(x, dim=1).unsqueeze(-1)
            entropies = norm / norm

        # graph convolution and probability scores
        probabilities = self.gc1(entropies, edge_index)
        probabilities = self.gc2(probabilities, edge_index)
        probabilities = self.gc3(probabilities, edge_index)
        probabilities = torch.sigmoid(probabilities)

        # conditional expectation; Algorithm 1
        gamma = entropies.sum()
        loss = self.loss_fn(entropies, probabilities, A, gamma)  # Eq. (9)

        mewis = self.conditional_expectation(entropies, probabilities, A, loss, gamma)

        # graph reconstruction; Eq. (10)
        x_pooled, adj_pooled = self.graph_reconstruction(mewis, x, A)
        edge_index_pooled, batch_pooled = self.to_edge_index(adj_pooled, mewis, batch)

        return x_pooled, edge_index_pooled, batch_pooled, loss, mewis

    @staticmethod
    def compute_entropy(x, L, A, batch):
        # computing local variations; Eq. (5)
        V = x * torch.matmul(L, x) - x * torch.matmul(A, x) + torch.matmul(A, x * x)
        V = torch.norm(V, dim=1)

        # computing the probability distributions based on the local variations; Eq. (7)
        P = torch.cat([torch.softmax(V[batch == i], dim=0) for i in torch.unique(batch)])
        P[P == 0.] += 1
        # computing the entropies; Eq. (8)
        H = -P * torch.log(P)

        return H.unsqueeze(-1)

    @staticmethod
    def loss_fn(entropies, probabilities, A, gamma):
        term1 = -torch.matmul(entropies.t(), probabilities)[0, 0]

        term2 = torch.matmul(torch.matmul(probabilities.t(), A), probabilities).sum()

        return gamma + term1 + term2

    def conditional_expectation(self, entropies, probabilities, A, threshold, gamma):
        sorted_probabilities = torch.sort(probabilities, descending=True, dim=0)

        dummy_probabilities = probabilities.detach().clone()
        selected = set()
        rejected = set()

        for i in range(sorted_probabilities.values.size(0)):
            node_index = sorted_probabilities.indices[i].item()
            neighbors = torch.where(A[node_index] == 1)[0]
            if len(neighbors) == 0:
                selected.add(node_index)
                continue
            if node_index not in rejected and node_index not in selected:
                s = dummy_probabilities.clone()
                s[node_index] = 1
                s[neighbors] = 0

                loss = self.loss_fn(entropies, s, A, gamma)

                if loss <= threshold:
                    selected.add(node_index)
                    for n in neighbors.tolist():
                        rejected.add(n)

                    dummy_probabilities[node_index] = 1
                    dummy_probabilities[neighbors] = 0

        mewis = list(selected)
        mewis = sorted(mewis)

        return mewis

    @staticmethod
    def graph_reconstruction(mewis, x, A):
        x_pooled = x[mewis]

        A2 = torch.matmul(A, A)
        A3 = torch.matmul(A2, A)

        A2 = A2[mewis][:, mewis]
        A3 = A3[mewis][:, mewis]

        I = torch.eye(len(mewis)).to('cuda:0')
        one = torch.ones([len(mewis), len(mewis)]).to('cuda:0')

        adj_pooled = (one - I) * torch.clamp(A2 + A3, min=0, max=1)

        return x_pooled, adj_pooled

    @staticmethod
    def to_edge_index(adj_pooled, mewis, batch):
        row1, row2 = torch.where(adj_pooled > 0)
        edge_index_pooled = torch.cat([row1.unsqueeze(0), row2.unsqueeze(0)], dim=0)
        batch_pooled = batch[mewis]

        return edge_index_pooled, batch_pooled


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Net, self).__init__()

        self.gc1 = GINConv(MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool1 = MEWISPool(hidden_dim=hidden_dim)
        self.gc2 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool2 = MEWISPool(hidden_dim=hidden_dim)
        self.gc3 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x, edge_index, batch):
        x = self.gc1(x, edge_index)
        x = torch.relu(x)

        x_pooled1, edge_index_pooled1, batch_pooled1, loss1, mewis = self.pool1(x, edge_index, batch)

        x_pooled1 = self.gc2(x_pooled1, edge_index_pooled1)
        x_pooled1 = torch.relu(x_pooled1)

        x_pooled2, edge_index_pooled2, batch_pooled2, loss2, mewis = self.pool2(x_pooled1, edge_index_pooled1,
                                                                                batch_pooled1)

        x_pooled2 = self.gc3(x_pooled2, edge_index_pooled2)

        readout = torch.cat([x_pooled2[batch_pooled2 == i].mean(0).unsqueeze(0) for i in torch.unique(batch_pooled2)],
                            dim=0)

        out = self.fc1(readout)
        out = torch.relu(out)
        out = self.fc2(out)

        return torch.log_softmax(out, dim=-1), loss1 + loss2


class Net2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Net2, self).__init__()

        self.gc1 = GINConv(MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.gc2 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.gc3 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool1 = MEWISPool(hidden_dim=hidden_dim)
        self.gc4 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x, edge_index, batch):
        x = self.gc1(x, edge_index)
        x = torch.relu(x)

        x = self.gc2(x, edge_index)
        x = torch.relu(x)

        x = self.gc3(x, edge_index)
        x = torch.relu(x)
        readout2 = torch.cat([x[batch == i].mean(0).unsqueeze(0) for i in torch.unique(batch)], dim=0)

        x_pooled1, edge_index_pooled1, batch_pooled1, loss1, mewis = self.pool1(x, edge_index, batch)

        x_pooled1 = self.gc4(x_pooled1, edge_index_pooled1)

        readout = torch.cat([x_pooled1[batch_pooled1 == i].mean(0).unsqueeze(0) for i in torch.unique(batch_pooled1)],
                            dim=0)

        out = readout2 + readout

        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)

        return torch.log_softmax(out, dim=-1), loss1


class Net3(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Net3, self).__init__()

        self.gc1 = GINConv(MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.gc2 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool1 = MEWISPool(hidden_dim=hidden_dim)
        self.gc3 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.gc4 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.pool2 = MEWISPool(hidden_dim=hidden_dim)
        self.gc5 = GINConv(MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, enhance=True))
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, x, edge_index, batch):
        x = self.gc1(x, edge_index).to('cuda:0')
        x = torch.relu(x)

        x = self.gc2(x, edge_index)
        x = torch.relu(x)

        x_pooled1, edge_index_pooled1, batch_pooled1, loss1, mewis1 = self.pool1(x, edge_index, batch)

        x_pooled1 = self.gc3(x_pooled1, edge_index_pooled1)
        x_pooled1 = torch.relu(x_pooled1)

        x_pooled1 = self.gc4(x_pooled1, edge_index_pooled1)
        x_pooled1 = torch.relu(x_pooled1)

        x_pooled2, edge_index_pooled2, batch_pooled2, loss2, mewis2 = self.pool2(x_pooled1, edge_index_pooled1,
                                                                                 batch_pooled1)

        x_pooled2 = self.gc5(x_pooled2, edge_index_pooled2)
        x_pooled2 = torch.relu(x_pooled2)

        readout = torch.cat([x_pooled2[batch_pooled2 == i].mean(0).unsqueeze(0) for i in torch.unique(batch_pooled2)],
                            dim=0)

        out = self.fc1(readout)
        out = torch.relu(out)
        out = self.fc2(out)

        return torch.log_softmax(out, dim=-1), loss1 + loss2
