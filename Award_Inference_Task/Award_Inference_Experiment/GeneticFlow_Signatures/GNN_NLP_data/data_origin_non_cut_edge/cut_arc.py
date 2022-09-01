import numpy as np
import os
import pandas as pd

import pickle
file = open('../data_with_proba/saved_model_another.pickle','rb')
model = pickle.load(file)
ratio=[]
for root, dirs, files in os.walk("../data_origin", topdown=False):
    for name in files:
        if(name[0]=='i'):
            df=pd.read_csv("../data_origin_with_feature/"+name,sep='\t',header=None)
            edge=df[[0,1]]
            df=df.drop(columns=[0,1])
            before=edge.shape[0]
            result=model.predict_proba(df)[:,0]
            y_pred = (result < 0.385).astype(int)
            Edge=[]
            df_paper=pd.read_csv("../csv/papers"+name[9:], header = None)
            df_paper=df_paper[df_paper[6].astype(float) >= 0.3]
            vertex=df_paper.iloc[:,0].values
            # print(vertex)
            for i in range(y_pred.shape[0]):
                if(y_pred[i]==1):
                    if (edge.iloc[i,0] in vertex) and (edge.iloc[i,1] in vertex):
                        # print([edge.iloc[i,0],edge.iloc[i,1]])
                        Edge.append([edge.iloc[i,0],edge.iloc[i,1]])
            
            ratio.append(len(Edge)/before)
            print(ratio[-1])
            np.savetxt(name, np.array(Edge), fmt="%s", delimiter="	")

print(np.mean(ratio))



