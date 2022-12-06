import pandas as pd
import re
import os
import shutil
import random
import numpy as np
# author_number=200

# count=0
# for root, dirs, files in os.walk(".", topdown=False):
#     for name in files:
#         if(name[0]=='l'):
#             number=re.findall(r"\d+\d*", name)[0]
#             number=int(number)
#             if(number <= author_number):
#                 count+=1
#                 shutil.copy(name,"../csv/")
# print(count)

# count=0
# for root, dirs, files in os.walk(".", topdown=False):
#     for name in files:
#         if(name[0]!='l' and name[-1]!='y' and name[0]!='t'):
#             number=re.findall(r"\d+\d*", name)[0]
#             number=int(number)
#             if(number <= author_number):
#                 count+=1
#                 shutil.copy(name,"../csv/")
# print(count)

# shutil.copy("top_field_authors.csv","../csv/")



# non_fellow=pd.read_csv('../../../../GeneticFlow_Signatures/GNN_NLP_data/non_fellow.csv',header=None).values
non_fellow=pd.read_csv('../../../../../GeneticFlow_Signatures/GNN_othersField/data/non_fellow.csv',header=None).values
non_fellow=non_fellow.reshape(-1).tolist()
# fellow=pd.read_csv('../../../../GeneticFlow_Signatures/GNN_NLP_data/fellow.csv',header=None).values
fellow=pd.read_csv('../../../../../GeneticFlow_Signatures/GNN_othersField/data/fellow.csv',header=None).values
fellow=fellow.reshape(-1).tolist()
print(non_fellow,len(non_fellow))
print(fellow,len(fellow))

count=0
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]!='l' and name[-1]!='y' and name[0]!='t'):
            number=re.findall(r"\d+\d*", name)[0]
            number=int(number)
            if(number in non_fellow):
                count+=1
                shutil.copy(name,"../csv/")
            elif(number in fellow):
                count+=1
                shutil.copy(name,"../csv/")
print(count)

count=0
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]=='l'):
            number=re.findall(r"\d+\d*", name)[0]
            number=int(number)
            if(number in non_fellow):
                count+=1
                shutil.copy(name,"../csv/")
            elif(number in fellow):
                count+=1
                shutil.copy(name,"../csv/")
print(count)

# shutil.copy("top_field_authors.csv","../csv/")
shutil.copy("top_field_authors.csv","../csv/")