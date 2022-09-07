import pandas as pd
import os
import sys
if len(sys.argv) < 2:
    print("Not enough parameters: ", len(sys.argv))
    sys.exit

number = sys.argv[1]

df=pd.read_csv("NLP_top_authors.csv")['name']
df=df.values[int(number):int(number)+10]
print(df)
for i in df:
    i=i.replace(" ","+")
    print('python Advisor-students.py '+ i)
    os.system('python Advisor-students.py '+ i)