import numpy as np
import os
import pandas as pd
from sklearn.impute import SimpleImputer
df_test = pd.read_csv('all_features.csv')
df_name=df_test[["citingpaperID","citedpaperID"]]
df_test=df_test.drop(columns=["citingpaperID","citedpaperID"])

# print(df_test.describe())
numpy_array = df_test.to_numpy()
imp = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=-2)
numpy_array = imp.fit_transform(numpy_array)
db2=pd.DataFrame(numpy_array)

db2=pd.concat([df_name,db2],axis=1)
print(db2)

for root, dirs, files in os.walk("../data_origin", topdown=False):
    for name in files:
        if(name[0]=='i'):
            arc=[]
            path="../data_origin/"+name
            edges_unordered = np.genfromtxt(path,dtype=np.dtype(str))
            if(edges_unordered.shape[0]==0):
                np.savetxt("../data_origin_with_feature/"+name, np.array(arc), fmt="%s", delimiter="	")
                continue
            if(len(edges_unordered.shape)==1):
                edges_unordered=edges_unordered[np.newaxis,:]
            for i in range(edges_unordered.shape[0]):
                occ=db2.loc[db2['citingpaperID']==edges_unordered[i][0]].loc[db2['citedpaperID']==edges_unordered[i][1]]
                if(len(occ.values)!=0):
                    allfeatures=occ.values[0].tolist()
                    arc.append(allfeatures)
                else:
                    arc.append([edges_unordered[i][0],edges_unordered[i][1],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    # arc.append([edges_unordered[i][0],edges_unordered[i][1],-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0,-2.0])
            print(name,len(arc))
            np.savetxt("../data_origin_with_feature/"+name, np.array(arc), fmt="%s", delimiter="	")

            
            