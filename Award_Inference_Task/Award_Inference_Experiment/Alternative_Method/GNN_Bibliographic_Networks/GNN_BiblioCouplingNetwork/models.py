import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, GlobalAttention, GraphMultisetTransformer, Set2Set, GlobalAttention
from torch_geometric.nn import ResGatedGraphConv,ChebConv,SAGEConv,GCNConv,GATConv,TransformerConv,AGNNConv,EdgePooling,GraphConv,GCN2Conv,TopKPooling,SAGPooling
from torch_geometric.nn import GINConv,GATv2Conv,ARMAConv
from torch_geometric.utils import get_laplacian
from layers import GCN, HGPSLPool

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

        # self.edgeconv = MLP(8, 64, 1).to('cuda:0')
        self.feature_mlp = MLP(6, 32, 32).to('cuda:0')
        
        # checkpoint=torch.load('mlp.pkl')
        # self.edgeconv.load_state_dict(checkpoint)

        # self.conv1 = ChebConv(self.num_features, self.nhid,1)
        # self.conv2 = ChebConv(self.nhid, self.nhid,1)
        # self.conv3 = ChebConv(self.nhid, self.nhid,1)
        self.conv1 = ARMAConv(self.num_features, self.nhid)
        self.conv2 = ARMAConv(self.nhid, self.nhid)
        self.conv3 = ARMAConv(self.nhid, self.nhid)
        # self.conv4 = GATv2Conv(self.nhid, self.nhid)
        # self.conv5 = GATv2Conv(self.nhid, self.nhid)

        # self.pool1 = SAGPooling(self.nhid, ratio=0.5, GNN=GATv2Conv)
        # self.pool2 = SAGPooling(self.nhid, ratio=0.5, GNN=GATv2Conv)
        # self.pool1 = TopKPooling(self.nhid, ratio=0.5)
        # self.pool2 = TopKPooling(self.nhid, ratio=0.5)
        # self.pool1 = EdgePooling(self.nhid, dropout=0.5, add_to_edge_score=0.5)
        # self.pool2 = EdgePooling(self.nhid, dropout=0.5, add_to_edge_score=0.5)

        # self.pool1 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        # self.pool2 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        # self.pool3 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        # self.pool4 = HGPSLPool(self.nhid, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        # self.globalpool = GlobalAttention(self.nhid, 10)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin2 = torch.nn.Linear(self.nhid // 2 + 1, self.nhid // 4)
        self.lin3 = torch.nn.Linear(self.nhid // 4, self.num_classes)

    def forward(self, data, paper_count):

        x, edge_index, batch= data.x.to('cuda:0'), data.edge_index.to('cuda:0'), data.batch.to('cuda:0')
        graph_ave_feature = gap(x, batch)
        edge_attr = data.edge_attr.to('cuda:0')

        # cos1=torch.flatten(edge_attr).to('cuda:0')
        # norm = torch.mean(cos1)
        # cos2=torch.flatten(edge_proba).to('cuda:0')
        # cos=cos1-cos2
        # print(torch.mean(cos1),torch.mean(cos2))
        # norm = torch.norm(cos, p=1, dim=0)
        # norm = norm/cos1.shape[0]


        x = F.relu(self.conv1(x.float(), edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = F.relu(self.conv1(x.float(), edge_index, edge_attr))
        # x, edge_index, batch, _ = self.pool1(x, edge_index, batch)
        # x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        # x, edge_index, edge_attr, batch = self.pool1(x, edge_index, None, batch)

        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x1 = gap(x, batch)

        x = F.relu(self.conv2(x, edge_index, None))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x, edge_index, batch, _ = self.pool2(x, edge_index, batch)
        # x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        # x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
        x2 = gap(x, batch)

        x = F.relu(self.conv3(x, edge_index, None))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x, edge_index, edge_attr, batch = self.pool3(x, edge_index, edge_attr, batch)
        x3 = gap(x, batch)
        
        # x = F.relu(self.conv4(x, edge_index))
        # x, edge_index, edge_attr, batch = self.pool4(x, edge_index, edge_attr, batch)
        # x4 = gap(x, batch)
        
        # x = F.relu(self.conv5(x, edge_index))
        # x5 = gap(x, batch)

        x = F.relu(x1) + F.relu(x3) + F.relu(x2)
        # x = F.relu(x1) + F.relu(x3) + F.relu(x2) + F.relu(x4) + F.relu(x5)
        # x = torch.cat((F.relu(x1), F.relu(x3)),1)

        # paper_count = (paper_count-torch.min(paper_count))/(torch.max(paper_count)-torch.min(paper_count))
        paper_count = torch.unsqueeze(paper_count,1)

        graph_ave_feature = torch.cat((graph_ave_feature, paper_count),1)
        # print(graph_ave_feature)
        # graph_ave_feature = self.feature_mlp(graph_ave_feature)

        # x = torch.cat((x, graph_ave_feature),1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        
        x = torch.cat((x, paper_count),1)
        x = F.relu(self.lin2(x))
        
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        # x = torch.cat((x, graph_ave_feature),1)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
