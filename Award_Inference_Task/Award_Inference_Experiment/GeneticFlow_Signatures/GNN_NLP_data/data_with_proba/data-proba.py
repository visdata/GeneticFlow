import numpy as np
import os
import pandas as pd

import pickle
# saved_model_another.pickle is for 36 features, and saved_model.pickle is for 9 features.
file = open('saved_model_another.pickle','rb')
model = pickle.load(file)

for root, dirs, files in os.walk("../data", topdown=False):
    for name in files:
        if(name[0]=='i'):
            df=pd.read_csv(name,sep='\t',header=None)
            edge=df[[0,1]]
            df=df.drop(columns=[0,1])
            result=model.predict_proba(df)[:,0]
            result=pd.DataFrame(result)
            # print(result)
            edge=pd.concat([edge,result],axis=1)
            print(edge.values.shape)
            np.savetxt(name, edge.values, fmt="%s", delimiter="	")



