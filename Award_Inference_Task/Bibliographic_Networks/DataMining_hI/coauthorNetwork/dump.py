import os
import re
import string
import json
import sys

if len(sys.argv) < 2:
    print("Not enough parameters: ", len(sys.argv))
    sys.exit

field_ = sys.argv[1]

host = "127.0.0.1"
port = "3306"
database = "scigene_"+field_+"_field_coauthorNetwork"
usr = "root"
pwd = "Vis_2014"

import pymysql

conn = pymysql.connect(host="127.0.0.1",user="root",password=pwd,db=database,autocommit=True)

cur = conn.cursor()

sql = "show tables"

cur.execute(sql)


for field in cur.fetchall():
    field=field[0]
    dumpfieldPapers = """select * from """ + field + """ INTO OUTFILE '/home/leishi/scigene/dataset/MAG/others_graph/""" + field_ + """/coauthorNetwork/""" + field + """.csv' FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\\n'"""
    cur.execute(dumpfieldPapers)

cur.close()

conn.close()