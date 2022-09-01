import pandas as pd
import os

def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

record=pd.DataFrame()
for root, dirs, files in os.walk("../data_origin", topdown=False):
    for name in files:
        if(name[0]=='l'):
            try:
                df=pd.read_csv(name,sep='	',header=None)
                record=pd.concat([record,df])
            except:
                continue
record=record.sort_values(by=2,ascending=False)
print(record)
record=record[2].values
import numpy as np 

import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
for i in range(10):
    print_to_file("record.txt",record[int(record.shape[0]/10*(i+1))-1])
    

# plt.plot(range(record.shape[0]), record, 'b-', linewidth=2)
# plt.savefig("a.png")