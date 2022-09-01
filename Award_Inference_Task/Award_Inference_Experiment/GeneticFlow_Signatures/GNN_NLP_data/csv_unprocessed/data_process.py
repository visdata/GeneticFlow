import pandas as pd
import re
import os
import shutil
import random
import numpy as np

fellow_number=50
non_fellow_number=150

def uniform_sample(all_items, sample_num):
    all_num = len(all_items)
    if sample_num > all_num or sample_num <= 0:
        print("Invalid sample: ", sample_num, " from ", all_num)
        return None
    sampled_items=[]
    sample_interval = (float(all_num)/sample_num)
    current_sample_index = 0
    for i in range(sample_num):
        sampled_items.append(all_items[int(current_sample_index)])
        current_sample_index+=sample_interval
        
    return sampled_items


authors = pd.read_csv("top_field_authors.csv", header = None)
fellow=[]
non_fellow=[]
total=[]

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]=='p'):
            number=re.findall(r"\d+\d*", name)[0]
            number=int(number)
            total.append(number)
print("total:",len(total))

fellow_count=0
count_till_Fellow=0
for i in range(authors.shape[0]):
    if(("2:" in authors.iloc[i,12] or "3:" in authors.iloc[i,12] or ("1:" in authors.iloc[i,12] and authors.iloc[i,5]>=3000)) and (authors.iloc[i][10] in total)):
        # fellow.append(authors.iloc[i][10])
        fellow_count+=1
        if(fellow_count==fellow_number):
            count_till_Fellow=i+1
            break

non_fellow_count=0
count_till_non_Fellow=0
for i in range(authors.shape[0]):
    if((not ("2:" in authors.iloc[i,12] or "3:" in authors.iloc[i,12] or ("1:" in authors.iloc[i,12] and authors.iloc[i,5]>=3000))) and (authors.iloc[i][10] in total)):
        # non_fellow.append(authors.iloc[i][10])
        non_fellow_count+=1
        if(non_fellow_count==non_fellow_number):
            count_till_non_Fellow=i+1
            break
print("count_till_Fellow:",count_till_Fellow)          
print("count_till_non_Fellow:",count_till_non_Fellow)
count=max(count_till_Fellow,count_till_non_Fellow)
print("count:",count)


for i in range(authors.shape[0]):
    if(("2:" in authors.iloc[i,12] or "3:" in authors.iloc[i,12] or ("1:" in authors.iloc[i,12] and authors.iloc[i,5]>=3000)) and (authors.iloc[i][10] in total)):
        fellow.append(authors.iloc[i][10])
    if(i+1==count):
        break

for i in range(authors.shape[0]):
    if((not ("2:" in authors.iloc[i,12] or "3:" in authors.iloc[i,12] or ("1:" in authors.iloc[i,12] and authors.iloc[i,5]>=3000))) and (authors.iloc[i][10] in total)):
        non_fellow.append(authors.iloc[i][10])
    if(i+1==count):
        break

print(non_fellow,len(non_fellow))
print(fellow,len(fellow))

non_fellow=uniform_sample(non_fellow,non_fellow_number)
fellow=uniform_sample(fellow,fellow_number)

print(non_fellow,len(non_fellow))
print(fellow,len(fellow))

np.savetxt("../fellow.csv", np.array(fellow), fmt="%s", delimiter=",")
np.savetxt("../non_fellow.csv", np.array(non_fellow), fmt="%s", delimiter=",")

count=0
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]=='p'):
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
        if(name[0]=='i'):
            number=re.findall(r"\d+\d*", name)[0]
            number=int(number)
            if(number in non_fellow):
                count+=1
                shutil.copy(name,"../csv/")
            elif(number in fellow):
                count+=1
                shutil.copy(name,"../csv/")
print(count)

shutil.copy("top_field_authors.csv","../csv/")
# shutil.copy("all_features.csv","../csv/")