import numpy as np
import os
import pandas as pd

db2 = pd.read_csv('all_features.csv')

flag=0

for root, dirs, files in os.walk("../data_origin", topdown=False):
    for name in files:
        if(name[0]=='i'):
            arc=[]
            path="../data_origin/"+name
            edges_unordered = np.genfromtxt(path,dtype=np.dtype(str))
            for i in range(edges_unordered.shape[0]):
                occ=db2.loc[db2['citingpaperID']==edges_unordered[i][0]].loc[db2['citedpaperID']==edges_unordered[i][1]]['pred']
                if(len(occ.values)==0):
                    arc.append([edges_unordered[i][0],edges_unordered[i][1],np.nan])
                else:
                    arc.append([edges_unordered[i][0],edges_unordered[i][1],occ.values[0]])
            print(name,len(arc))
            if(flag==0):
                df_feature = pd.DataFrame(arc, columns=['citingpaperID', 'citedpaperID', 'extends_prob'])
                flag=1
            else:
                df_arc = pd.DataFrame(arc, columns=['citingpaperID', 'citedpaperID', 'extends_prob'])
                df_feature = pd.concat([df_feature, df_arc])
            print(df_feature)
            np.savetxt("../data_origin/"+name, np.array(arc), fmt="%s", delimiter="	")

df_feature.to_csv('df_feature.csv')
# from sqlalchemy import create_engine
# import sqlalchemy

# engine = create_engine('mysql+pymysql://root:Vis_2014@localhost:3306/scigene_acl_anthology')

# df_feature.to_sql('paper_reference_NLP_labeled', engine, if_exists='replace', index=False, dtype={"citingpaperID": sqlalchemy.types.NVNLPHAR(length=100),\
#     "citedpaperID": sqlalchemy.types.NVNLPHAR(length=100)})

            
            