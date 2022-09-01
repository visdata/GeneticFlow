
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

author_number=250

def node_iter(G):
   return G.nodes

def node_dict(G):
    node_dict = G.nodes
    return node_dict

def read_graphfile(PATH):
    node_num=1000
    adj_list={i:[] for i in range(1,node_num+1)}    
    index_graph={i:[] for i in range(1,node_num+1)}
    node_attrs={i:[] for i in range(1,node_num+1)} 
    graph_labels=[]
    graph_hindex=[]
    edge_weight=[]
    edge_proba=[]
    authors_attributes=[]
    Name=[]
    num_edges = 0
    index_i = 1
    for_i = -1
    # for root, dirs, files in os.walk("./data/data_with_proba", topdown=False):
    # for root, dirs, files in os.walk("./data/data_with_feature", topdown=False):
    # for root, dirs, files in os.walk("./data/data_origin", topdown=False):
    # for root, dirs, files in os.walk("./data/data", topdown=False):
    for root, dirs, files in os.walk(PATH, topdown=False):
        for name in files:
            if(name[0]=='p'):
                for_i += 1
                
                path=PATH+'/'+name
                # print(index_i,name)
                idx_features_labels = np.genfromtxt("{}".format(path),
                                    dtype=np.dtype(str))
                # build graph
                try:
                    idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
                    idx_map = {j: i for i, j in enumerate(idx)}
                except:
                    continue
                edges_unordered = np.genfromtxt(path.replace("papers", "links"),
                                                dtype=np.dtype(str))
                                                
                authors = pd.read_csv("./data/csv/"+"top_field_authors.csv", header = None)
                number=re.findall(r"\d+\d*", name)[0]
                number=int(number)
                fellow=authors[authors[6]==number][9].values[0]
                if(fellow!='\\N'):
                    if(('1:' in fellow) and (fellow.count(',')==1) and (authors[authors[6]==number][7].values[0]<1000)):
                        print('jump: ',fellow)
                        continue
                    templist=[]
                    templist.append(1)
                    graph_labels.append(templist)
                else:
                    templist=[]
                    templist.append(0)
                    graph_labels.append(templist)

                if(edges_unordered.shape[0]==0):
                    adj_list[index_i]=[]
                    edge_weight.append([])
                else:
                    if(len(edges_unordered.shape)==1):
                        edges_unordered=edges_unordered[np.newaxis,:]
                    # print(edges_unordered)
                    # print(edges_unordered.shape)
                    edgeset=set()
                    edges_=[]
                    # print(edges_unordered)
                    for j in range(edges_unordered.shape[0]):
                        if((edges_unordered[j][0],edges_unordered[j][1]) in edgeset or (edges_unordered[j][1],edges_unordered[j][0]) in edgeset):
                            continue
                        else:
                            if(edges_unordered[j][0]==edges_unordered[j][1]):
                                continue
                            # if(edges_unordered[j][2].astype(np.float32)>0.5):
                            edgeset.add((edges_unordered[j][0],edges_unordered[j][1]))
                            edges_.append(edges_unordered[j])
                    edges_unordered=np.array(edges_)
                    if(edges_unordered.shape[0]==0):
                        continue
                    edge_w_temp = edges_unordered[:,2:].astype(np.float32)
                    # print(edge_w_temp.shape[0],edges_unordered.shape[0])
                    edge_w=[]
                    for j in range(edge_w_temp.shape[0]):
                        edge_w.append(edge_w_temp[j,:])
                        edge_w.append(edge_w_temp[j,:])
                    edge_w=np.array(edge_w).astype(np.float32)
                    # print(edge_w,edge_w_temp)
                    edges_unordered = edges_unordered[:,:2]
                    edges_unordered = edges_unordered.astype(float).astype(int).astype(str)
                    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                                    dtype=np.int32).reshape(edges_unordered.shape)
                    edge_weight.append(edge_w)
                    for line in edges:
                        e0,e1=(int(line[1]),int(line[0]))
                        adj_list[index_i].append([e0,e1])
                        adj_list[index_i].append([e1,e0])
                        num_edges += 1

                fea_labels=pd.DataFrame(idx_features_labels)
                fea_labels[1]=pd.to_numeric(fea_labels[1])-1965
                fea_labels=fea_labels[[0,1,3,4,6,7,8]]
                idx_features_labels=fea_labels.values
                for line in idx_features_labels[:, 1:]:
                    attrs = [float(attr) for attr in line]
                    node_attrs[index_i].append(np.array(attrs))
                # authors2 = pd.read_csv('top_1000_graph_feature_keyNodeGraph_extendlink50-yuankai.csv')
                # authors2 = authors2.drop(columns=['authorID','authorACLFellow','authorACMFellow','sum_paperCitationMaxYearSpan','avg_paperCitationMaxYearSpan','sum_paperCitationHalfLifeYearSpan','avg_paperCitationHalfLifeYearSpan'])
                # authors2 = authors2.drop(columns=['authorID','authorACLFellow','authorACMFellow','sum_paperCitationMaxYearSpan','avg_paperCitationMaxYearSpan','sum_paperCitationHalfLifeYearSpan','avg_paperCitationHalfLifeYearSpan','sum_paperCitation','edgeSize','hIndexGraph','sum_avgOutDegreeCitation','sum_paperCitation-avgOutDegreeYearspan'])
                # authors2 = (authors2-authors2.mean())/authors2.std()
                # authors2 = pd.read_csv('../diffpool_and_sklearn/non_graph.csv')
                # authors2 = (authors2-authors2.mean())/authors2.std()
                
                Name.append(name)
                index_i+=1
                

    # sleep()
    return edge_weight,graph_labels,adj_list,node_attrs,Name
    # return graph_labels,adj_list,node_attrs
    # return edge_weight,graph_labels,adj_list,node_attrs
# T.ToDense(300)
class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, PATH, transform=None, pre_transform=None):
        self.PATH = PATH
        transform=T.Compose([
            # T.NormalizeFeatures(),
            # T.ToSparseTensor(),
            # T.ToSparseTensor(attr='edge_attr'),
        ])
        # super(MultiSessionsGraph, self).__init__(root, transform=T.ToSparseTensor(attr='edge_proba'))
        super(MultiSessionsGraph, self).__init__(root, transform=transform)
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
        edge_weight,y,adj_list,node_attrs,Name=read_graphfile(PATH=self.PATH)
        # y,adj_list,node_attrs=read_graphfile()
        # edge_weight,y,adj_list,node_attrs=read_graphfile()
        # print(edge_weight[0].shape)
        # sleep()
        for i in range(len(y)):
            number = re.findall(r"\d+\.?\d*",Name[i])
            number=int(number[0])
            pyg_graph=Data(x=torch.tensor(node_attrs[i+1],dtype=torch.float), y=torch.tensor(y[i],dtype=torch.float), edge_index=torch.tensor(adj_list[i+1],dtype=torch.long).t(),edge_attr=torch.tensor(edge_weight[i],dtype=torch.float),name=torch.tensor(number,dtype=torch.int))
            # pyg_graph=Data(x=torch.tensor(node_attrs[i+1],dtype=torch.float), y=torch.tensor(y[i],dtype=torch.long), edge_index=torch.tensor(adj_list[i+1],dtype=torch.long).t())
            # pyg_graph=Data(x=torch.tensor(node_attrs[i+1],dtype=torch.float), y=torch.tensor(y[i],dtype=torch.long), edge_index=torch.tensor(adj_list[i+1],dtype=torch.long).t(),edge_attr=torch.tensor(edge_weight[i],dtype=torch.float))
            print(pyg_graph)
            data_list.append(pyg_graph)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])