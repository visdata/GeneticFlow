import pandas as pd
import re
import os
import shutil
import random
import numpy as np
import sys
if len(sys.argv) < 2:
    print("Not enough parameters: ", len(sys.argv))
    sys.exit
    
if(sys.argv[1]=='NLP'):
    fellow = range(201) 
    count=0
    for root, dirs, files in os.walk("../../Award_Inference_Task/GeneticFlow_Graphs/NLP_ARC_hI/", topdown=False):
        for name in files:
            if(name[0]=='p'):
                number=re.findall(r"\d+\d*", name)[0]
                number=int(number)
                if (number in fellow):
                    shutil.copy("../../Award_Inference_Task/GeneticFlow_Graphs/NLP_ARC_hI/"+name,"./graph/")
    print(count)
else:
    field = sys.argv[1]
    fellow = range(201)

    count=0
    for root, dirs, files in os.walk("../../Award_Inference_Task/GeneticFlow_Graphs/"+field+"_hI/", topdown=False):
        for name in files:
            if(name[0]=='p'):
                number=re.findall(r"\d+\d*", name)[0]
                number=int(number)
                if (number in fellow):
                    shutil.copy("../../Award_Inference_Task/GeneticFlow_Graphs/"+field+"_hI/"+name,"./graph/")
    print(count)