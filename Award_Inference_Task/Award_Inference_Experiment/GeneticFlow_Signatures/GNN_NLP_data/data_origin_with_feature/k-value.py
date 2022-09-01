import pandas as pd
import os
import pickle
file = open('../data_with_proba/saved_model.pickle','rb')
# file = open('../data_with_proba/saved_model_another.pickle','rb')
model = pickle.load(file)
def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

record=pd.DataFrame()
for root, dirs, files in os.walk("../data_origin", topdown=False):
    for name in files:
        if(name[0]=='i'):
            try:
                df=pd.read_csv(name,sep='	',header=None)
                df=df.drop_duplicates()
                record=pd.concat([record,df])
            except:
                continue

print(record)
df2=record[[0,1]]
df=record.drop(columns=[0,1])
print(df)
result=model.predict_proba(df)[:,0]
record=pd.DataFrame(result)
print(record)
df2=df2.reset_index()
df2=df2[[0,1]]
df2=pd.concat([df2,record],axis=1)
df2.to_csv('edge_with_proba.csv',index=None,header=None)
record=record.sort_values(by=0,ascending=False)
# print(record)
record=record[0].values

import numpy as np 

import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
# print(record)
for i in range(10):
    print_to_file("record.txt",record[int(record.shape[0]/10*(i+1))-1])
    

plt.plot(range(record.shape[0]), record, 'b-', linewidth=2)
plt.savefig("a.png")