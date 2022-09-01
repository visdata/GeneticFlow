# -*- coding: utf-8 -*-

import pandas as pd
# from sqlalchemy import create_engine

# engine = create_engine('mysql+pymysql://root:Vis_2014@localhost:3306/scigene_acl_anthology')
# engine2 = create_engine('mysql+pymysql://root:Vis_2014@localhost:3306/scigene_features')
# sql = '''select * from papers_NLP;'''
# sql2 = '''select * from feature_66;'''

# db = pd.read_sql_query(sql, engine)
# db2 = pd.read_sql_query(sql2, engine2)

import numpy as np
import pandas as pd
import os
import re
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]=='l'):
            # print(name)
            # vertex=pd.read_csv(name, header = None)
            # # print(vertex)
            # vertex=vertex[vertex[6].astype(float) >= 0.5]
            # vertex=vertex.iloc[:,[0,2,3,4,5,6,7]]
            
            # vertex=vertex.replace(-1, np.nan)

            # vertex=vertex.fillna(vertex.mean())
            # # print(vertex)
            # # print(re.findall(r"\d+\d*",name))
            
            # vertex[7]=vertex[2]-vertex[2].min()
            # print(vertex)
            # vertex=vertex.values

            # np.savetxt("../data/"+name[:-4]+"_new.csv", np.array(vertex), fmt="%s", delimiter="	")
            
            try:
                arc=pd.read_csv(name, header = None)
            except:
                continue
            arc=arc.iloc[:,0:]
            arc=arc.values
            df=pd.read_csv("papers"+name[5:], header = None)
            df=df[df[6].astype(float) >= 0.5]
            # print(df)
            vertex=df.iloc[:,0].values
            arc_new=[]
            for s in arc:
                if (s[0] in vertex) and (s[1] in vertex):
                    # print(s)
                    citing=s[0]
                    cited=s[1]
            #         year_citing=db.loc[db['paperID']==s[0]]['year'].values[0]
            #         year_cited=db.loc[db['paperID']==s[1]]['year'].values[0]
            #         num_citing=db.loc[db['paperID']==s[0]]['citationCount'].values[0]
            #         num_cited=db.loc[db['paperID']==s[1]]['citationCount'].values[0]
            #         num_OCCURRENCES=db2.loc[db2['citingpaperID']=='2020.cl-2.6'].loc[db2['citedpaperID']=='W19-6102']['NUM_OCCURRENCES'].values[0]
                    if(s[3]=='\\N'):
                        arc_new.append([str(s[0]),str(s[1])])
                    else:
                        arc_new.append([str(s[0]),str(s[1])])
            #         print([s[0],s[1],year_citing-year_cited,num_citing,num_cited,num_OCCURRENCES])
            # print(len(arc_new))
            # sleep()
            np.savetxt("../data/"+name[:-4]+"_new.csv", np.array(arc_new), fmt="%s", delimiter="	")