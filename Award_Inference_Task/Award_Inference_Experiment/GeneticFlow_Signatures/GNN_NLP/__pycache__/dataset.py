
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
import joblib
import scipy as sc

def node_iter(G):
   return G.nodes

def node_dict(G):
    node_dict = G.nodes
    return node_dict

def read_graphfile(PATH,union_field=False,train_all=False):
    pklname = os.path.basename(PATH) + '-allfield.pkl'
    if train_all:
        try:
            edge_weight,graph_labels,adj_list,node_attrs,Name = joblib.load(pklname)
            return edge_weight,graph_labels,adj_list,node_attrs,Name
        except:
            raise f"not found {pklname}"
    sub_graph_num=1000
    adj_list={i:[] for i in range(1,sub_graph_num+1)}     #adjacent info:{sub_graph_index_i: [[e0,e1],[e1,e0]]}
    index_graph={i:[] for i in range(1,sub_graph_num+1)}
    node_attrs={i:[] for i in range(1,sub_graph_num+1)}  #node vector
    graph_labels=[]
    graph_hindex=[]
    edge_weight=[] #edge vector:{sub_graph_index_i:[atrr(e0,e1),atrr(e1,e0)])
    edge_proba=[]
    Name=[]
    authors_attributes=[]
    num_edges = 0
    index_i = 1
    for_i = -1
    for root, dirs, files in os.walk(PATH, topdown=False):
        for name in files:
            if(name[0]=='p'):
                for_i += 1
                path=PATH+'/'+name
                idx_features_labels = np.genfromtxt("{}".format(path),
                                    dtype=np.dtype(str))
                # build graph
                idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
                idx_map = {j: i for i, j in enumerate(idx)}
                edges_unordered = np.genfromtxt(path.replace("papers", "influence"),
                                                dtype=np.dtype(str))
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
                            edgeset.add((edges_unordered[j][0],edges_unordered[j][1]))
                            edges_.append(edges_unordered[j])
                    edges_unordered=np.array(edges_)
                    # print(edges_unordered)
                    if(edges_unordered.shape[0]==0):
                        adj_list[index_i]=[]
                        edge_weight.append([])
                    else:
                        edge_w_temp = edges_unordered[:,2:].astype(np.float32)
                        edge_w=[]
                        for j in range(edge_w_temp.shape[0]):
                            edge_w.append(edge_w_temp[j,:])
                            edge_w.append(edge_w_temp[j,:])
                        edge_w=np.array(edge_w).astype(np.float32)
                        # print(edge_w,edge_w_temp)
                        edges_unordered = edges_unordered[:,:2]
                        # print(edges_unordered,idx_map)
                        # print(edges_unordered)
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
                fea_labels=fea_labels[[0,1,3,4,6,8,9]]
                # docx=pd.read_csv("../gensim/similarity_features.csv")
                # for i in range(fea_labels.shape[0]):
                #     if(len(docx[docx['paperID']==fea_labels.iloc[i,0]].values)!=0):
                #         fea_labels.iloc[i,5]=docx[docx['paperID']==fea_labels.iloc[i,0]].iloc[0,3]
                #         fea_labels.iloc[i,6]=docx[docx['paperID']==fea_labels.iloc[i,0]].iloc[0,4]
                #     else:
                #         fea_labels.iloc[i,5]=0
                #         fea_labels.iloc[i,6]=0
                idx_features_labels=fea_labels.values
                for line in idx_features_labels[:, 1:]:
                    attrs = [float(attr) for attr in line]
                    node_attrs[index_i].append(np.array(attrs))

                authors = pd.read_csv("../GNN_NLP_data/csv/top_field_authors.csv", header = None)
                number=re.findall(r"\d+\d*", name)[0]
                number=int(number)

                if("2:" in authors[authors[10]==number][12].values[0] or "3:" in authors[authors[10]==number][12].values[0] or ("1:" in authors[authors[10]==number][12].values[0] and authors[authors[10]==number][5].values[0]>=3000)):
                    # templist=[float(attr) for attr in authors2.iloc[for_i][1:].values]
                    templist=[]
                    templist.append(1)
                    graph_labels.append(templist)
                else:
                    # templist=[float(attr) for attr in authors2.iloc[for_i][1:].values]
                    templist=[]
                    templist.append(0)
                    graph_labels.append(templist)

                Name.append(name)
                index_i+=1

    # sleep()
    if union_field:
        try:
            fielddata = joblib.load(pklname)
        except:
            fielddata=[]
        fielddata.append([edge_weight,graph_labels,adj_list,node_attrs,Name])
        joblib.dump(fielddata, pklname)
    return edge_weight,graph_labels,adj_list,node_attrs,Name

# T.ToDense(300)
import re
class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, PATH, train_all, transform=None, pre_transform=None):
        self.PATH = PATH
        self.train_all=train_all
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
        edge_weight,y,adj_list,node_attrs,Name=read_graphfile(PATH=self.PATH,train_all=self.train_all)
        
        for i in range(len(y)):
            number = re.findall(r"\d+\d*",Name[i])
            number=int(number[0])
            pyg_graph=Data(x=torch.tensor(node_attrs[i+1],dtype=torch.float), y=torch.tensor(y[i],dtype=torch.float), edge_index=torch.reshape(torch.tensor(adj_list[i+1],dtype=torch.long).t(),(2,-1)),edge_attr=torch.tensor(edge_weight[i],dtype=torch.float),name=torch.tensor(number,dtype=torch.int))
            # print(pyg_graph.x.shape,pyg_graph.y.shape,pyg_graph.edge_index.shape,pyg_graph.edge_attr.shape,pyg_graph.name.shape)
            data_list.append(pyg_graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])