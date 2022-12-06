# coding=gbk
import time
import sys
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from fake_useragent import UserAgent
from selenium.webdriver.common.by import By
from dateutil.parser import parse

from gensim import utils
import re
import gensim
from gensim.parsing.preprocessing import preprocess_string
gensim.parsing.preprocessing.STOPWORDS = set()
import time
import sys

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

def init_option():
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-gpu')
    options.add_argument('--hide-scrollbars')
    options.add_argument('blink-settings=imagesEnabled=false')
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument("--disable-javascript")
    ua = UserAgent()
    options.add_argument('user-agent="%s"' % ua.random)
    # options.add_experimental_option('excludeSwitches', ['enable-automation'])
    return options

driver = webdriver.Chrome(options=init_option())

if len(sys.argv) < 2:
    print("Not enough parameters: ", len(sys.argv))
    sys.exit

topauthor = sys.argv[1]
print_to_file("Non_student_"+topauthor+".txt",topauthor+",min_year,max_year")
print_to_file(topauthor+".txt",topauthor+",min_year,max_year")
driver.get('https://openreview.net/search?term='+topauthor+'&group=all&content=authors&source=forum')
driver.implicitly_wait(5)

Student=[]
Non_Student=[]
already_go=[]
Non_Student_year={}
def is_advisor(author,year):
    Advisor=[]
    All_Advisor=[]
    min_year=10000
    max_year=-1
    min_year_phd=10000
    max_year_phd=-1
    driver.implicitly_wait(10)
    student_before=author.text
    author.click()
    driver.implicitly_wait(3)
    try:
        advisor=driver.find_elements(By.XPATH,"/html/body/div/div[3]/div/div/main/div/div/div/section[5]/div/div")
        student=driver.find_element(By.XPATH,"//*[@id=\"content\"]/div/header/div/h1").text
        student=student_before
    except:
        driver.back()
        return
    for i in advisor:
        try:
            if("PhD Advisor" in i.find_element(By.XPATH,'./div[1]').text):
                Advisor.append(i.find_element(By.XPATH,'./div[2]').text)
                print(i.find_element(By.XPATH,'./div[2]').text+" is Advisor of "+student)
            if("Advisor" in i.find_element(By.XPATH,'./div[1]').text):
                All_Advisor.append(i.find_element(By.XPATH,'./div[2]').text)
                Advisor_year=i.find_element(By.XPATH,'./div[4]').text
                Advisor_year=Advisor_year.replace("Present", "2022")
                print(i.find_element(By.XPATH,'./div[2]').text+" is Advisor of "+student+" in "+Advisor_year)
                Advisor_year=Advisor_year.split(" – ")
                min_year=min(min_year,int(Advisor_year[0]))
                max_year=max(max_year,int(Advisor_year[0]))
                min_year=min(min_year,int(Advisor_year[1]))
                max_year=max(max_year,int(Advisor_year[1]))
                if("PhD Advisor" in i.find_element(By.XPATH,'./div[1]').text):
                    min_year_phd=min(min_year_phd,int(Advisor_year[0]))
                    max_year_phd=max(max_year_phd,int(Advisor_year[0]))
                    min_year_phd=min(min_year_phd,int(Advisor_year[1]))
                    max_year_phd=max(max_year_phd,int(Advisor_year[1]))
        except:
            print("No")
    flag=0
    Non_Student_year[student]=(min_year,max_year)
    for name in All_Advisor:
        if(is_same_author(topauthor,name)):
            flag=1
    if(len(All_Advisor)!=0 and flag==0 and year>=min_year and year<=max_year and (not is_same_author(topauthor,student))):
        print(All_Advisor,min_year,max_year,student)
        Non_Student.append(student)
    for name in Advisor:
        if(is_same_author(topauthor,name)):
            if(student not in Student):
                Student.append(student)
                print_to_file(topauthor+".txt",student+","+str(min_year_phd)+","+str(max_year_phd))
            print(student+" is student of "+topauthor)
            break
    print(student)
    driver.back()

try:
    while(1):
        count=0
        for i in range(25):
            driver.implicitly_wait(5)
            paper=driver.find_elements(By.XPATH,"/html/body/div/div[3]/div/div/main/div/div/ul/li["+str(i+1)+"]/div/div/a")
            year=driver.find_element(By.XPATH,"/html/body/div/div[3]/div/div/main/div/div/ul/li["+str(i+1)+"]/div/ul/li[1]").text
            year=year.split('(')[0]
            try:
                year=parse(year, fuzzy=True).year
            except:
                continue
            for idx,val in enumerate(paper):
                driver.implicitly_wait(20)
                try:
                    j=driver.find_element(By.XPATH,"/html/body/div/div[3]/div/div/main/div/div/ul/li["+str(i+1)+"]/div/div/a["+str(idx+1)+"]")
                    author=j.get_attribute("href")
                    if("dblp.org" in author):
                        continue
                    print(author)
                    if(author not in already_go):
                        is_advisor(j,year)
                        already_go.append(author)
                    else:
                        if(j.text in Non_Student):
                            print(j.text+" is in non_student, check it!")
                            print(Non_Student)
                            min_year=Non_Student_year[j.text][0]
                            max_year=Non_Student_year[j.text][1]
                            if(year>=min_year and year<=max_year):
                                Non_Student.remove(j.text)
                            print(Non_Student)
                except:
                    print("skip")
                    continue
                count+=1
        print(count)
        time.sleep(2)
        driver.find_element(By.XPATH,"/html/body/div/div[3]/div/div/main/div/div/nav/ul/li[13]/a").click()
except:
    for i in Non_Student:
        print_to_file("Non_student_"+topauthor+".txt",i+","+str(Non_Student_year[i][0])+","+str(Non_Student_year[i][1]))
    print(already_go)
    driver.quit()