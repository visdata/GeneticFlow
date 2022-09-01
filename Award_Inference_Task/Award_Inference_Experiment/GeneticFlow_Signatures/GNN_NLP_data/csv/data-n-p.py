# -*- coding: utf-8 -*-

import pandas as pd
# from sqlalchemy import create_engine

import numpy as np
import pandas as pd
import os
import re
ratio=[]
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]=='p'):
            print(name)
            vertex=pd.read_csv(name, header = None)
            before=vertex.shape[0]
            vertex=vertex[vertex[6].astype(float) < 0.3]
            ratio.append(vertex.shape[0]/before)
            print(ratio[-1])
            vertex=vertex.iloc[:,[0,2,3,4,5,6,7,10,12,13]]
            
            vertex=vertex.replace(-1, np.nan)

            vertex=vertex.fillna(vertex.mean())
            vertex[7]=vertex[2]-vertex[2].min()
            vertex=vertex.values

            np.savetxt("../non_data/"+name, np.array(vertex), fmt="%s", delimiter="	")

            
print(np.mean(ratio))