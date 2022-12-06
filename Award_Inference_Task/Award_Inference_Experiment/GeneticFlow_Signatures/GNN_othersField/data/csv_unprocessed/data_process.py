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

try:
    arcmatched = np.load('../arc2mag_nlp.npz')
    match_flag = True
except:
    match_flag = False

authors = pd.read_csv("top_field_authors.csv", header = None)
fellow=[]
non_fellow=[]
total=[]

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[0]=='p'):
            number=re.findall(r"\d+\d*", name)[0]
            number=int(number)
            if number<=220 or not match_flag:
                total.append(number)
print("total:",len(total))

fellow_count=0
count_till_Fellow=0
for i in range(authors.shape[0]):
    if(authors.iloc[i][9]!='\\N' and (authors.iloc[i][6] in total)):
        if(('1:' in authors.iloc[i][9]) and (authors.iloc[i][9].count(',')==1) and (authors.iloc[i][7]<1000)):
            print('jump: ',authors.iloc[i][9],authors.iloc[i][7])
            continue
    # if(authors.iloc[i][9]!='\\N' and (authors.iloc[i][6] in total)):
        # fellow.append(authors.iloc[i][10])
        fellow_count+=1
        if(fellow_count==fellow_number):
            count_till_Fellow=i+1
            break

non_fellow_count=0
count_till_non_Fellow=0
print(authors)
for i in range(authors.shape[0]):
    if(authors.iloc[i][9]=='\\N' and (authors.iloc[i][6] in total)):
    # if(authors.iloc[i][9]=='\\N'):
        # non_fellow.append(authors.iloc[i][10])
        non_fellow_count+=1
        # print(non_fellow_count)
        if(non_fellow_count==non_fellow_number):
            count_till_non_Fellow=i+1
            break
print("count_till_Fellow:",count_till_Fellow)          
print("count_till_non_Fellow:",count_till_non_Fellow)
count=max(count_till_Fellow,count_till_non_Fellow)
print("count:",count)


for i in range(authors.shape[0]):
    if(authors.iloc[i][9]!='\\N' and (authors.iloc[i][6] in total)):
        if(('1:' in authors.iloc[i][9]) and (authors.iloc[i][9].count(',')==1) and (authors.iloc[i][7]<1000)):
            print('jump: ',authors.iloc[i][9],authors.iloc[i][7])
            continue
    # if(authors.iloc[i][9]!='\\N' and (authors.iloc[i][6] in total)):
        fellow.append(authors.iloc[i][6])
    if(i+1==count):
        break

for i in range(authors.shape[0]):
    if(authors.iloc[i][9]=='\\N' and (authors.iloc[i][6] in total)):
    # if(authors.iloc[i][9]=='\\N'):
        non_fellow.append(authors.iloc[i][6])
    if(i+1==count):
        break

print(non_fellow,len(non_fellow))
print(fellow,len(fellow))

non_fellow=uniform_sample(non_fellow,non_fellow_number)
fellow=uniform_sample(fellow,fellow_number)


if match_flag:
    print('find ../arc2mag_nlp.npz, start mapping sampling')
    matched_fellow = list(arcmatched['matched_magfellow'])
    matched_nonfellow = list(arcmatched['matched_magnonfellow'])
    Topmatched_author = list(arcmatched['Topmatched_mag'])
    topmatched_rest=set(Topmatched_author)-set(matched_nonfellow)-set(matched_fellow)
    lack_fellow=len(fellow)-len(matched_fellow)
    for ele in topmatched_rest:
        if lack_fellow <= 0:
            break
        if ele not in matched_fellow and ele not in matched_nonfellow and ele in fellow:
            matched_fellow.append(ele)
            lack_fellow-=1
    for add_f in fellow:
        if lack_fellow <= 0:
            break
        if add_f not in matched_fellow and add_f not in matched_nonfellow:
            matched_fellow.append(add_f)
            lack_fellow-=1
    lack_nonfellow=len(non_fellow)-len(matched_nonfellow)
    for ele in topmatched_rest:
        if lack_nonfellow <= 0:
            break
        if ele not in matched_fellow and ele not in matched_nonfellow and ele in non_fellow:
            matched_nonfellow.append(ele)
            lack_nonfellow-=1
    for add_nf in non_fellow:
        if lack_nonfellow <= 0:
            break
        if add_nf not in matched_fellow and add_nf not in matched_nonfellow:
            matched_nonfellow.append(add_nf)
            lack_nonfellow-=1
    fellow=matched_fellow
    non_fellow=matched_nonfellow
print()
print('after sampling...')
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


# count=0
# for root, dirs, files in os.walk(".", topdown=False):
#     for name in files:
#         if(name[0]=='p'):
#             number=re.findall(r"\d+\d*", name)[0]
#             number=int(number)
#             if(number <= author_number):
#                 count+=1
#                 shutil.copy(name,"../csv/")
# print(count)

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

shutil.copy("top_field_authors.csv","../csv/")
# shutil.copy("all_features.csv","../csv/")