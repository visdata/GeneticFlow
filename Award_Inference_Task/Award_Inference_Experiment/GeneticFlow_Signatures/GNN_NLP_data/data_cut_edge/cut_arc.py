import numpy as np
import os
import pandas as pd

import pickle
file = open('../data_with_proba/saved_model_another.pickle','rb')
model = pickle.load(file)
ratio=[]
for root, dirs, files in os.walk("../data", topdown=False):
    for name in files:
        if(name[0]=='i'):
            df=pd.read_csv(name,sep='\t',header=None)
            edge=df[[0,1]]
            df=df.drop(columns=[0,1])
            before=edge.shape[0]
            result=model.predict_proba(df)[:,0]
            y_pred = (result > 0.321).astype(int)
            Edge=[]
            for i in range(y_pred.shape[0]):
                if(y_pred[i]==1):
                    # print([edge.iloc[i,0],edge.iloc[i,1]])
                    Edge.append([edge.iloc[i,0],edge.iloc[i,1]])
            ratio.append(len(Edge)/before)
            print(ratio[-1])
            # np.savetxt(name, np.array(Edge), fmt="%s", delimiter="	")

print(np.mean(ratio))



