import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import sys
if len(sys.argv) < 2:
    print("Not enough parameters: ", len(sys.argv))
    sys.exit

database = sys.argv[1]

engine = create_engine('mysql+pymysql://root:Vis_2014@localhost:3306/'+database)


sql_paper = '''select * from all_dataset_link_with_features; '''
db_data=pd.read_sql_query(sql_paper, engine)
db_data['citingpaperID']=db_data['citingpaperID'].astype(str)
db_data['citedpaperID']=db_data['citedpaperID'].astype(str)
db_data.to_csv('all_features.csv',index=False)