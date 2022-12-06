import pandas as pd
import re
import os
import shutil
import random
import numpy as np
# author_number=200

# non_fellow=pd.read_csv('../../../../../GeneticFlow_Signatures/GNN_NLP_data/non_fellow.csv',header=None).values
non_fellow_num=pd.read_csv('../non_fellow.csv',header=None).values
non_fellow_num=non_fellow_num.reshape(-1).tolist()
# fellow=pd.read_csv('../../../../../GeneticFlow_Signatures/GNN_NLP_data/fellow.csv',header=None).values
fellow_num=pd.read_csv('../fellow.csv',header=None).values
fellow_num=fellow_num.reshape(-1).tolist()
print(non_fellow_num,len(non_fellow_num))
print(fellow_num,len(fellow_num))
count=0

# samplelist=os.listdir(".")
# samplelist=list(filter(lambda x: "papers" in x, samplelist))
# fellow={}
# non_fellow={}
# for tablename in samplelist:
#     number=re.findall(r"\d+\d*", tablename)[0]
#     number=int(number)
#     authorname=re.findall(r"_+(.*)%s"%number,tablename)[0].split('_')[-1]
#     if(number in fellow_num):
#         fellow[authorname]=str(number)
#     elif(number in non_fellow_num):
#         non_fellow[authorname]=str(number)

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]!='l' and name[-1]!='y' and name[0]!='t'):
            number=re.findall(r"\d+\d*", name)[0]
            number=int(number)
            authorname=re.findall(r"_+(.*)%s"%number,name)[0].split('_')[-1]
            if(number in non_fellow_num):
                count+=1
                shutil.copy(name,"../csv/")
            elif(number in fellow_num):
                count+=1
                shutil.copy(name,"../csv/")
print(count)
count=0
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]=='l'):
            number=re.findall(r"\d+\d*", name)[0]
            number=int(number)
            authorname=re.findall(r"_+(.*)%s"%number,name)[0].split('_')[-1]
            if(number in non_fellow_num):
                count+=1
                shutil.copy(name,"../csv/")
            elif(number in fellow_num):
                count+=1
                shutil.copy(name,"../csv/")
print(count)

# shutil.copy("top_field_authors.csv","../csv/")
shutil.copy("top_field_authors.csv","../csv/")