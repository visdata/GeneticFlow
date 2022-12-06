import time
import os
from dateutil.parser import parse
from sqlalchemy import create_engine
import sys
import pandas as pd

if len(sys.argv) < 2:
    print("Not enough parameters: ", len(sys.argv))
    sys.exit

field = sys.argv[1]

numOfTopAuthor=500
if len(sys.argv) > 2:
    numOfTopAuthor=sys.argv[2]

def factorial(n):
    return 1 if (n==1 or n==0) else n * factorial(n - 1)
count=0
casual=0
if field.lower()=="acl":
    engine = create_engine('mysql+pymysql://root:Vis_2014@localhost:3306/scigene_acl_anthology')
    sql = f'''select distinct paperID from authors_ARC_hI,paper_author_ARC where authorRank <= {numOfTopAuthor} and paper_author_ARC.authorID=authors_ARC_hI.authorID;'''
else:
    engine = create_engine('mysql+pymysql://root:Vis_2014@localhost:3306/scigene_'+field+'_field')
    sql = f'''select distinct paperID from authors_field_hI,paper_author_field  where authorRank <= {numOfTopAuthor} and paper_author_field.authorID=authors_field_hI.authorID;'''
# print(field,numOfTopAuthor)
db = pd.read_sql_query(sql, engine).values
for i in range(db.shape[0]):
    paperID = db[i,0]
    if field.lower()=="acl":
        sql = '''select name,authorOrder,title,papers_ARC.paperID from paper_author_ARC,authors_ARC_hI,papers_ARC where paper_author_ARC.paperID=papers_ARC.paperID and paper_author_ARC.paperID='''+'\''+paperID+'\' and paper_author_ARC.authorID=authors_ARC_hI.authorID order by authorOrder;'
    else:
        sql = '''select name,authorOrder,title,papers_field.paperID from paper_author_field,authors_field_hI,papers_field where paper_author_field.paperID=papers_field.paperID and paper_author_field.paperID='''+'\''+paperID+'\' and paper_author_field.authorID=authors_field_hI.authorID  order by authorOrder;'
    authororder = pd.read_sql_query(sql, engine).values
    # print(authororder)
    authororder = authororder[:,0]
    authororder = [str.split()[-1] for str in authororder]
    if(all(authororder[str] <= authororder[str+1] for str in range(len(authororder) - 1))):
        count += 1
    if(len(authororder)>=10):
        casual += 0
    else:
        casual += 1.0/factorial(len(authororder))
        # print(1.0/factorial(len(authororder)),len(authororder))
    # print(authororder,all(authororder[str] <= authororder[str+1] for str in range(len(authororder) - 1)))
# print(count/db.shape[0])
# print(casual/db.shape[0])
# print(db.shape[0])
print('Total number of papers:',db.shape[0])
print('Number of alphabetical sequence papers:',count)
print('Real Frequency: %.3f'%(count/db.shape[0]))
print('Theoretical probability: %.3f'%(casual/db.shape[0]))
print('diff: %.3f'%(count/db.shape[0]-casual/db.shape[0]))