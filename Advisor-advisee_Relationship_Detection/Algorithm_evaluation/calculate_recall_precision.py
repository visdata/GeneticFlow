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

tp=0
fp=0
fn=0
tn=0

for root, dirs, files in os.walk("./graph", topdown=False):
    for name in files:
        if(name[0]=='p'):
            print(name)
            number=re.findall(r"\d+\d*", name)[0]
            number=int(number)
            
            df=pd.read_csv("./graph/"+name,header=None)
            df=df[[6,8]]
            df=df.groupby([8])[8,6].agg(['max']) #df[8]='firstAuthorName' df[6]='advisor-advisee-rate of firstAuthor'
            Advisor_advisee_rate=df.values
            # print(Advisor_advisee_rate[:,0])
            
            students=pd.read_csv("../Advisor_PhD_crawler/student/"+str(number)+".txt",header=None)
            students=students.iloc[1:,:].values
            if(students.shape[0]==0):
                continue
            if(students.shape[0]>=1):
                students=students.reshape(-1)
            for first in students:
                for idx in range(Advisor_advisee_rate.shape[0]):
                    second=Advisor_advisee_rate[idx,0]
                    # print(first,second)
                    if(is_same_author(first,second)):
                        if(Advisor_advisee_rate[idx,1]>=0.5):
                            tp+=1
                        else:
                            fn+=1
                        break

            non_students=pd.read_csv("../Advisor_PhD_crawler/non_student/"+str(number)+".txt",header=None)
            non_students=non_students.iloc[1:,:].drop_duplicates().values
            if(non_students.shape[0]==0):
                continue
            if(non_students.shape[0]>=1):
                non_students=non_students.reshape(-1)
            for first in non_students:
                for idx in range(Advisor_advisee_rate.shape[0]):
                    second=Advisor_advisee_rate[idx,0]
                    # print(first,second)
                    if(is_same_author(first,second)):
                        # print(first,Advisor_advisee_rate[idx])
                        if(Advisor_advisee_rate[idx,1]>=0.5):
                            fp+=1
                        else:
                            tn+=1
                        break
                        
print("tp=",tp,"fp=",fp,"fn=",fn,"tn=",tn)
print("positive=",tp+fn,"negative=",fp+tn)
precision=tp/(tp+fp)
recall=tp/(tp+fn)

print("recall: ", tp/(tp+fn))
print("precision: ", tp/(tp+fp))
print("f1:",2*precision*recall/(recall+precision))

# 1.Initial samples
# tp= 287 fp= 38 fn= 64 tn= 131
# positive= 351 negative= 169
# recall:  0.8176638176638177
# precision:  0.8830769230769231
# f1: 0.8491124260355029

# 2.Delete PhD students in the positive samples and those who have only collaborated with their supervisors on one dissertation
# tp= 287 fp= 38 fn= 41 tn= 131
# positive= 328 negative= 169
# recall:  0.875
# precision:  0.8830769230769231
# f1: 0.8790199081163859
