import os
import pandas as pd
from gensim import utils
import re
import gensim
from gensim.parsing.preprocessing import preprocess_string
gensim.parsing.preprocessing.STOPWORDS = set()
import time
import sys
import numpy as np
import shutil

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

df=pd.read_csv("../top_field_authors.csv")['name'].head(200).values

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if(name[-1]=='t'):
            author=name.split(".txt")[0].split("Non_student_")[1].replace("+", " ")
            print(author)
            number=np.where(df==author)[0][0]+1
            shutil.copy(name,str(number)+".txt")