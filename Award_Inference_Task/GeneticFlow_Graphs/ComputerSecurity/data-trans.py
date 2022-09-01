# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np
import pandas as pd
import os
import re
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]=='l'):
            # print(name)
            try:
                arc=pd.read_csv(name, header = None)
            except:
                print(name)
                os.system('rm -f '+name)
                os.system('rm -f '+"papers"+name[5:])
                # np.savetxt("./"+"new_"+name, np.array(arc_new), fmt="%s", delimiter="	")
                continue
            arc=arc.iloc[:,0:]
            arc=arc.values
            df=pd.read_csv("papers"+name[5:], header = None)
            check_df=df[df[6].astype(float) >= 0.5]
            # print(df)
            vertex=df.iloc[:,0].values
            arc_new=[]
            for s in arc:
                if (s[0] in vertex) and (s[1] in vertex):
                    # print(s)
                    citing=int(s[0])
                    cited=int(s[1])
            #         year_citing=db.loc[db['paperID']==s[0]]['year'].values[0]
            #         year_cited=db.loc[db['paperID']==s[1]]['year'].values[0]
            #         num_citing=db.loc[db['paperID']==s[0]]['citationCount'].values[0]
            #         num_cited=db.loc[db['paperID']==s[1]]['citationCount'].values[0]
            #         num_OCCURRENCES=db2.loc[db2['citingpaperID']=='2020.cl-2.6'].loc[db2['citedpaperID']=='W19-6102']['NUM_OCCURRENCES'].values[0]
                    if(s[3]=='\\N'):
                        arc_new.append([str(int(s[0])),str(int(s[1])),0.5])
                    else:
                        arc_new.append([str(int(s[0])),str(int(s[1])),s[3]])
            #         print([s[0],s[1],year_citing-year_cited,num_citing,num_cited,num_OCCURRENCES])
            # print(arc_new)
            # sleep()
            np.savetxt("./"+"new_"+name, np.array(arc_new), fmt="%s", delimiter="	")