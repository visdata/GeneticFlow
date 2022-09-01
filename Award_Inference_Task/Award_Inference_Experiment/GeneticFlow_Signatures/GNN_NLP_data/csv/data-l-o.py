# -*- coding: utf-8 -*-

import pandas as pd
# from sqlalchemy import create_engine

import numpy as np
import pandas as pd
import os
import re
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]=='i'):
            print(name)
            arc=pd.read_csv(name, header = None)
            arc=arc.iloc[:,0:]
            arc=arc.values
            df=pd.read_csv("papers"+name[9:], header = None)
            vertex=df.iloc[:,0].values
            arc_new=[]
            for s in arc:
                if (s[0] in vertex) and (s[1] in vertex):
                    # print(s)
                    citing=s[0]
                    cited=s[1]
                    if(s[3]=='\\N'):
                        arc_new.append([s[0],s[1],0.5])
                    else:
                        arc_new.append([s[0],s[1],s[3]])
            print(len(arc_new))
            np.savetxt("../data_origin/"+name, np.array(arc_new), fmt="%s", delimiter="	")