import numpy as np
import scipy as sc
import os
import re
import pandas as pd

node_attrs={i:[] for i in range(200)} 
node_attrs2={i:[] for i in range(200)} 
graph_labels=[]
Name=[]
index_i=0
# for root, dirs, files in os.walk("./data/data_origin", topdown=False):
for root, dirs, files in os.walk("../GNN_NLP_data/data_origin", topdown=False):
    for name in files:
        if(name[0]=='p'):
            # path="./data/data_origin/"+name
            path="../GNN_NLP_data/data_origin/"+name
            idx_features_labels = np.genfromtxt("{}".format(path),
                                    dtype=np.dtype(str))
            
            idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
            # print(index_i,len(idx_features_labels))
            features=pd.DataFrame(idx_features_labels)
            features=features.replace('-1', np.nan)
            features=features.dropna(axis=0)
            idx_features_labels=features.values
            for line in idx_features_labels[:, [1,3,4,7,8,9]]:
                attrs = [float(attr) for attr in line]
                # print(attrs)
                # print(np.array(attrs))
                node_attrs[index_i].append(np.array(attrs))
            node_attrs[index_i]=np.mean(node_attrs[index_i],axis=0).tolist()
            node_attrs[index_i].append(idx_features_labels.shape[0])
            node_attrs[index_i]=np.array(node_attrs[index_i])
            # print(node_attrs[index_i])
            
            # authors2 = pd.read_csv('top_1000_graph_feature_keyNodeGraph_extendlink50-yuankai.csv')
            # authors2 = authors2.drop(columns=['authorID','authorACLFellow','authorACMFellow','sum_paperCitationMaxYearSpan','avg_paperCitationMaxYearSpan','sum_paperCitationHalfLifeYearSpan','avg_paperCitationHalfLifeYearSpan','sum_paperCitation','avg_paperCitation','nodeSize','edgeSize','hIndexGraph','sum_avgOutDegreeCitation','sum_paperCitation-avgOutDegreeYearspan','sum_paperCitation-maxOutDegreeYearspan'])
            # authors2 = authors2.drop(columns=['authorID','authorACLFellow','authorACMFellow','sum_paperCitationMaxYearSpan','avg_paperCitationMaxYearSpan','sum_paperCitationHalfLifeYearSpan','avg_paperCitationHalfLifeYearSpan'])
            authors = pd.read_csv("../GNN_NLP_data/csv/top_field_authors.csv", header = None)
            number=re.findall(r"\d+\d*", name)[0]
            number=int(number)
            Name.append(number)
            if("2:" in authors[authors[10]==number][12].values[0] or "3:" in authors[authors[10]==number][12].values[0] or ("1:" in authors[authors[10]==number][12].values[0] and authors[authors[10]==number][5].values[0]>=3000)):
                graph_labels.append(1)
            else:
                graph_labels.append(0)
            authors[5] = authors[5]/authors[4]
            authors = authors[[4,5,6,10]]
            node_attrs2[index_i]=authors[authors[10]==number][[4,5,6]].values[0]
            print(name,graph_labels[index_i],node_attrs[index_i])
            index_i+=1
            
            

df=pd.DataFrame({"graph_labels":graph_labels,"name":Name})

df2=pd.DataFrame(node_attrs2)
df2=df2.T
df2.columns=['PaperCount_ARC','CitationCount_ARC','hIndex_ARC']
# df2.columns=['PaperCount_ARC','CitationCount_ARC','hIndex_ARC','PaperCount_ACL']
# df2.columns=['authorCitationNLP','authorHIndexNLP','componentSize','hIndexhComponent']
# df2.columns=['authorCitationNLP','authorHIndexNLP','sum_paperCitation','avg_paperCitation','sum_avgOutDegreeCitation','sum_paperCitation-avgOutDegreeYearspan','sum_paperCitation-maxOutDegreeYearspan','nodeSize','edgeSize','componentSize','hIndexGraph','hIndexhComponent']

df1=pd.DataFrame(node_attrs)
df1=df1.T
df1.columns=['year','ave_citation','ave_authorOrder','ave_year_sub_first','paper_number','x1','x2']
df=pd.concat([df, df2], axis=1)
print(df)
df.to_csv('non_graph.csv',index=False)