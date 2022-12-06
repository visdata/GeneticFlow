import os
import re
import string
import json
import sys

host = "127.0.0.1"
port = "3306"
database = "scigene_acl_anthology_coauthorNetwork"
usr = "root"
pwd = "Vis_2014"

import pymysql

conn = pymysql.connect(host="127.0.0.1",user="root",password=pwd,db=database,autocommit=True)

cur = conn.cursor()

sql = "show tables"

cur.execute(sql)


for field in cur.fetchall():
    field=field[0]
    dumpfieldPapers = """select * from """ + field + """ INTO OUTFILE '/home/leishi/scigene/dataset/MAG/others_graph/NLP_ARC/coauthorNetwork/""" + field + """.csv' FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\\n'"""
    cur.execute(dumpfieldPapers)

cur.close()

conn.close()