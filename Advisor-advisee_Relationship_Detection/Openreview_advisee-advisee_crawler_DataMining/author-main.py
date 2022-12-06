import pandas as pd
import os
import sys
if len(sys.argv) < 2:
    print("Not enough parameters: ", len(sys.argv))
    sys.exit

number = sys.argv[1]

df=pd.read_csv("top_field_authors.csv")['name']
df=df.values[int(number)*2:int(number)*2+20]
print(df)
for i in df:
    i=i.replace(" ","+")
    print('python author.py '+ i)
    os.system('python author.py '+ i)