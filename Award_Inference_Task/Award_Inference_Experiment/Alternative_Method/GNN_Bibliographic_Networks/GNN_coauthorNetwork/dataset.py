
import torch
from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx
import torch_geometric.transforms as T
import os
import re
import pandas as pd
import scipy as sc

def node_iter(G):
   return G.nodes

def node_dict(G):
    node_dict = G.nodes
    return node_dict

def read_graphfile(max_nodes=None):
    node_num=1000
    adj_list={i:[] for i in range(1,node_num+1)}    
    index_graph={i:[] for i in range(1,node_num+1)}
    node_attrs={i:[] for i in range(1,node_num+1)} 
    graph_labels=[]
    graph_hindex=[]
    edge_weight=[]
    edge_proba=[]
    Name=[]
    authors_attributes=[]
    num_edges = 0
    index_i = 1
    for_i = -1
    for root, dirs, files in os.walk("./data/csv", topdown=False):
    # for root, dirs, files in os.walk("../GNN_NLP_data/data_with_feature", topdown=False):
    # for root, dirs, files in os.walk("../GNN_NLP_data/data_with_proba", topdown=False):
        for name in files:
            if(name[0]!='l' and name[0]!='t'):
                for_i += 1
                
                path="./data/csv/"+name

                idx_features_labels = pd.read_csv(path,header=None).values
                # build graph
                idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
                idx_map = {j: i for i, j in enumerate(idx)}
                try:
                    edges_unordered = pd.read_csv('./data/csv/links_'+name.split('_')[1],header=None).values
                except:
                    continue
                edgeset=set()
                edges_=[]
                for j in range(edges_unordered.shape[0]):
                    if(edges_unordered[j][0]==edges_unordered[j][1]):
                        continue
                    # if(edges_unordered[j][2].astype(np.float32)>0.5):
                    edgeset.add((edges_unordered[j][0],edges_unordered[j][1]))
                    edges_.append(edges_unordered[j])

                edges_unordered=np.array(edges_)
                # print(edges_unordered)
                # s()
                edge_w = edges_unordered[:,2:].astype(np.float32)
                edge_w=edge_w.reshape(-1)
                if((np.max(edge_w)-np.min(edge_w))!=0):
                    edge_w=(edge_w-np.min(edge_w))/(np.max(edge_w)-np.min(edge_w))
                else:
                    edge_w=edge_w
                edge_w=edge_w.reshape(-1,1)
                # print(edge_w)
                # s()
                edges_unordered = edges_unordered[:,:2]
                edges_unordered=edges_unordered.astype(int).astype(str)
                edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                                dtype=np.int32).reshape(edges_unordered.shape)
                # print(edges)
                # s()
                fea_labels=pd.DataFrame(idx_features_labels)
                # print(fea_labels)
                # s()
                fea_labels=fea_labels[[0,2,3,4]]
                fea_labels=fea_labels.replace('\\N',-1)
                fea_labels[4]=fea_labels[4].astype(int)
                fea_labels=fea_labels.replace(-1, np.nan)
                fea_labels=fea_labels.fillna(fea_labels.mean())
                idx_features_labels=fea_labels.values
                # print(fea_labels.mean())
                for line in edges:
                    e0,e1=(int(line[0]),int(line[1]))
                    adj_list[index_i].append([e0,e1])
                    num_edges += 1
                for line in idx_features_labels[:, 1:]:
                    # print(line)
                    attrs = [float(attr) for attr in line]
                    node_attrs[index_i].append(np.array(attrs))
                
                # authors = pd.read_csv("./data/csv/top_field_authors.csv", header = None)
                authors = pd.read_csv("./data/csv/top_field_authors.csv", header = None)
                number=re.findall(r"\d+\d*", name)[0]
                number=int(number)
                
                # if("1:" in authors[authors[10]==number][12].values[0] or "3:" in authors[authors[10]==number][12].values[0] or ("2:" in authors[authors[10]==number][12].values[0] and authors[authors[10]==number][5].values[0]>=3000)):
                if(authors[authors[6]==number][9].values[0]!='\\N'):
                    # templist=[float(attr) for attr in authors2.iloc[for_i][1:].values]
                    templist=[]
                    templist.append(1)
                    graph_labels.append(templist)
                    # print(number,1)
                else:
                    # templist=[float(attr) for attr in authors2.iloc[for_i][1:].values]
                    templist=[]
                    templist.append(0)
                    graph_labels.append(templist)
                # print(name,templist,authors.iloc[number-1,12])
                edge_weight.append(edge_w)
                Name.append(name)
                index_i+=1

    # sleep()
    return edge_weight,graph_labels,adj_list,node_attrs,Name
    # return graph_labels,adj_list,node_attrs
    # return edge_weight,graph_labels,adj_list,node_attrs
# T.ToDense(300)
import re
class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, transform=None, pre_transform=None):
        transform=T.Compose([
            # T.NormalizeFeatures(),
            # T.ToSparseTensor(),
            # T.ToSparseTensor(attr='edge_attr'),
        ])
        super(MultiSessionsGraph, self).__init__(root, transform=transform)
        # super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return ['data.txt']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass
    
    def process(self):
        
        data_list = []
        edge_weight,y,adj_list,node_attrs,Name=read_graphfile()
        # y,adj_list,node_attrs=read_graphfile()
        # edge_weight,y,adj_list,node_attrs=read_graphfile()
        # print(edge_weight[0].shape)
        # sleep()
        for i in range(len(y)):
            number = re.findall(r"\d+\d*",Name[i])
            number=int(number[0])
            pyg_graph=Data(x=torch.tensor(node_attrs[i+1],dtype=torch.float), y=torch.tensor(y[i],dtype=torch.float), edge_index=torch.tensor(adj_list[i+1],dtype=torch.long).t(),edge_attr=torch.tensor(edge_weight[i],dtype=torch.float),name=torch.tensor(number,dtype=torch.int))
            # pyg_graph=Data(x=torch.tensor(node_attrs[i+1],dtype=torch.float), y=torch.tensor(y[i],dtype=torch.long), edge_index=torch.tensor(adj_list[i+1],dtype=torch.long).t())
            # pyg_graph=Data(x=torch.tensor(node_attrs[i+1],dtype=torch.float), y=torch.tensor(y[i],dtype=torch.long), edge_index=torch.tensor(adj_list[i+1],dtype=torch.long).t(),edge_attr=torch.tensor(edge_weight[i],dtype=torch.float))
            print(pyg_graph)
            data_list.append(pyg_graph)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])