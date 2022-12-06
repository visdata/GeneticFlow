import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
import sys
import re

from gensim import utils
import gensim
from gensim.parsing.preprocessing import preprocess_string
gensim.parsing.preprocessing.STOPWORDS = set()
import time
from sqlalchemy import create_engine

if len(sys.argv) < 3:
    print("Not enough parameters: ", len(sys.argv))
    sys.exit

field = sys.argv[1]
ratio_PAA = float(sys.argv[2])

def strip_short2(s, minsize=2):
    s = utils.to_unicode(s)
    def remove_short_tokens(tokens, minsize):
        return [token for token in tokens if len(token) >= minsize]
    return " ".join(remove_short_tokens(s.split(), minsize))
gensim.parsing.preprocessing.DEFAULT_FILTERS[6]=strip_short2
del gensim.parsing.preprocessing.DEFAULT_FILTERS[-1]

def is_same_author(first,second):
    first=set(preprocess_string(first))
    second=set(preprocess_string(second))
    x = first.intersection(second)
    if(len(x)>=2 or (len(x)==1 and len(first)==1 and len(second)==1)):
        return True
    else:
        return False

def print_to_file(filename, string_info, mode="a"):
    # i=0
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

tp=0
fp=0
fn=0
tn=0

pd.set_option('display.max_rows',None)
for root, dirs, files in os.walk("./graph", topdown=False):
    for name in files:
        if(name[0]=='p'):
            print(name)
            number=re.findall(r"\d+\d*", name)[0]
            number=int(number)
            
            df=pd.read_csv("./graph/"+name,header=None)
            df=df[[8,6,2]] #df[8]='firstAuthorName' df[6]='advisor-advisee-ratio of firstAuthor'
            df=df.groupby([8,2])[8,6,2].agg(['max']) #df[8]='firstAuthorName' df[6]='advisor-advisee-ratio of firstAuthor'
            # print(df)
            Advisor_advisee_ratio=df.values
            # print(Advisor_advisee_ratio[:,0])
            
            print_to_file("positive_samples.txt",name.split(".")[0]+":")
            print_to_file("negative_samples.txt",name.split(".")[0]+":")
            students_pd=pd.read_csv("../Openreview_advisee-advisee_crawler_"+field+"/student/"+str(number)+".txt",header=None)
            students=students_pd.iloc[1:,0].values
            min_year=students_pd.iloc[1:,1].values
            max_year=students_pd.iloc[1:,2].values
            if(students.shape[0]==0):
                continue
            if(students.shape[0]>=1):
                students=students.reshape(-1)
                min_year=min_year.reshape(-1)
                max_year=max_year.reshape(-1)
            for idx_first, first in enumerate(students):
                for idx in range(Advisor_advisee_ratio.shape[0]):
                    second=Advisor_advisee_ratio[idx,0]
                    year=int(Advisor_advisee_ratio[idx,2])
                    # print(first,second)
                    if(is_same_author(first,second) and year>=int(min_year[idx_first]) and year<=int(max_year[idx_first])):
                        
                        print_to_file("positive_samples.txt",first)
                        if(Advisor_advisee_ratio[idx,1]>=ratio_PAA):
                            tp+=1
                        else:
                            print(first," phd ",Advisor_advisee_ratio[idx])
                            # print('python check_advisor_advisee.py '+ "'"+Advisor_advisee_ratio[idx,0]+"' "+str(number))
                            # os.system('python check_advisor_advisee.py '+ "'"+Advisor_advisee_ratio[idx,0]+"' "+str(number))
                            fn+=1
                        break

            non_students=pd.read_csv("../Openreview_advisee-advisee_crawler_"+field+"/non_student/"+str(number)+".txt",header=None)
            min_year=non_students.iloc[1:,1].values
            max_year=non_students.iloc[1:,2].values
            non_students=non_students.iloc[1:,0].drop_duplicates().values
            if(non_students.shape[0]==0):
                continue
            if(non_students.shape[0]>=1):
                non_students=non_students.reshape(-1)
                min_year=min_year.reshape(-1)
                max_year=max_year.reshape(-1)
            for idx_first, first in enumerate(non_students):
                for idx in range(Advisor_advisee_ratio.shape[0]):
                    second=Advisor_advisee_ratio[idx,0]
                    year=int(Advisor_advisee_ratio[idx,2])
                    # print(first,second)
                    if(is_same_author(first,second) and year>=int(min_year[idx_first]) and year<=int(max_year[idx_first])):
                        print_to_file("negative_samples.txt",first)
                        # print(first,Advisor_advisee_ratio[idx])
                        if(Advisor_advisee_ratio[idx,1]>=ratio_PAA):
                            # print(first,Advisor_advisee_ratio[idx],year,min_year[idx_first],max_year[idx_first])
                            # os.system('python check_advisor_advisee.py '+ "'"+Advisor_advisee_ratio[idx,0]+"' "+str(number))
                            fp+=1
                        else:
                            tn+=1
                        break
print("tp=",tp,"fp=",fp,"fn=",fn,"tn=",tn)
print("positive=",tp+fn,"negative=",fp+tn)
precision=tp/(tp+fp)
recall=tp/(tp+fn)

def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")
print_to_file("./graph/record.txt","Precision:%.3f Recall:%.3f F1:%.3f ACC:%.3f for Class Fellow" % (tp/(tp+fp),tp/(tp+fn),2*precision*recall/(recall+precision),(tp+tn)/(tp+tn+fp+fn)))
print("recall: ", tp/(tp+fn))
print("precision: ", tp/(tp+fp))
print("f1:",2*precision*recall/(recall+precision))
print("acc:",(tp+tn)/(tp+tn+fp+fn))
