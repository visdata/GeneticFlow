import os
import re
import string
import json
import math
import sys

# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(curPath)

from odbcdb import *
import pandas as pd
import numpy as NUMTOPAUTHORS
import matplotlib.pyplot as plt
import seaborn as sns

def draw_distribution_histogram(nums, path, is_hist=True, is_kde=True, is_rug=False, \
  is_vertical=False, is_norm_hist=False):
  
  sns.set()  
  sns.distplot(nums, bins=100, hist=is_hist, kde=is_kde, rug=is_rug, \
    hist_kws={"color":"steelblue"}, kde_kws={"color":"purple"}, \
    vertical=is_vertical, norm_hist=is_norm_hist)

  plt.xlabel("X")
  plt.ylabel("Y")

  plt.title("Distribution")
  plt.tight_layout()  
  plt.savefig(path, dpi=300)


MIN_SUPERVISED_YEAR_SPAN = 2
MIN_SUPERVISED_PAPER_SPAN = 2

numOfTopAuthors = 500

if len(sys.argv) >= 2:
    numOfTopAuthors = int(sys.argv[1])

if len(sys.argv) >= 4:
    MIN_SUPERVISED_YEAR_SPAN = float(sys.argv[2])
    MIN_SUPERVISED_PAPER_SPAN = float(sys.argv[3])

MIN_STUDENT_AUTHOR_ORDER = 3

MIN_SUPERVISOR_RATE = 0.01

MIN_SUPERVISED_RATE = 1
MIN_SUPERVISING_RATE = 1

MAX_SUPERVISED_YEAR = 6
HALF_SUPERVISED_YEAR = 3
MAX_YEAR = 1000

MAX_SUPERVISED_PAPER = 10
HALF_SUPERVISED_PAPER = 5
MAX_PAPER = 1000

MAX_ACADEMIC_YEAR = int(
    MAX_SUPERVISED_YEAR
    - 1
    - math.log(MIN_SUPERVISOR_RATE * MIN_SUPERVISED_RATE)
    * HALF_SUPERVISED_YEAR
    / math.log(2)
)

SUPERVISED_YEAR_MODIFIER = []

for i in range(MAX_YEAR):
    if i < MAX_SUPERVISED_YEAR:
        SUPERVISED_YEAR_MODIFIER.append(1)
    else:
        SUPERVISED_YEAR_MODIFIER.append(
            math.exp(
                -math.log(2) * (i - MAX_SUPERVISED_YEAR + 1) / HALF_SUPERVISED_YEAR
            )
        )

SUPERVISED_PAPER_MODIFIER = []

for i in range(MAX_PAPER):
    if i < MAX_SUPERVISED_PAPER:
        SUPERVISED_PAPER_MODIFIER.append(1)
    else:
        SUPERVISED_PAPER_MODIFIER.append(
            math.exp(
                -math.log(2) * (i - MAX_SUPERVISED_PAPER + 1) / HALF_SUPERVISED_PAPER
            )
        )


def computeSupervisorRate(
    studentID,
    supervisorID,
    year,
    firstAuthorPaperCountMap,
    firstAuthorWeightedPaperCountMap,
    coAuthorWeightedPaperCountMap,
    coAuthorPaperCountMap,
    topAuthorPaperCountMap,
):

    # compute supervised rate

    studentPaperCountMap = firstAuthorPaperCountMap[studentID]

    # the sorted list of years that the student has paper publication, truncated to {0,1,..., MAX_ACADEMIC_YEAR}
    studentAcademicYearList = sorted(studentPaperCountMap.keys())[
        0 : MAX_ACADEMIC_YEAR + 1
    ]
    max_student_academic_year = len(studentAcademicYearList) - 1

    if not (year in studentAcademicYearList):
        return 0.0, 0.0, 0.0

    currentAcademicYearIndex = studentAcademicYearList.index(year)

    studentWeightedPaperCountMap = firstAuthorWeightedPaperCountMap[studentID]

    coAuthorID = studentID + "-" + supervisorID
    studentCoAuthorWeightedPaperCountMap = coAuthorWeightedPaperCountMap[coAuthorID]

    # the same as below except that co-author weighted count is replaced by student weighted count

    start_student_count_list = {}
    end_student_count_list = {}
    total_student_count = 0

    start_student_count_list[0] = 0
    for i in range(1, max_student_academic_year + 1):
        if studentAcademicYearList[i - 1] in studentWeightedPaperCountMap:
            start_student_count_list[i] = (
                start_student_count_list[i - 1]
                + studentWeightedPaperCountMap[studentAcademicYearList[i - 1]]
            )
        else:
            start_student_count_list[i] = start_student_count_list[i - 1]

    end_student_count_list[max_student_academic_year] = 0
    for i in range(max_student_academic_year - 1, currentAcademicYearIndex - 1, -1):
        if studentAcademicYearList[i + 1] in studentWeightedPaperCountMap:
            end_student_count_list[i] = (
                end_student_count_list[i + 1]
                + studentWeightedPaperCountMap[studentAcademicYearList[i + 1]]
            )
        else:
            end_student_count_list[i] = end_student_count_list[i + 1]

    total_student_count = (
        start_student_count_list[currentAcademicYearIndex]
        + end_student_count_list[currentAcademicYearIndex]
        + studentWeightedPaperCountMap[
            studentAcademicYearList[currentAcademicYearIndex]
        ]
    )

    # start_list[N] = accumulated weighted co-author paper count from academic year 0 to N-1, excluding the year N, N <= current_academic_year
    # end_list[N] = accumulated weighted co-author paper count from academic year N to MAX_ACADEMIC_YEAR, excluding the year N, N >= current_academic_year

    start_coauthor_count_list = {}
    end_coauthor_count_list = {}
    total_coauthor_count = 0

    start_coauthor_count_year_list = {}
    end_coauthor_count_year_list = {}
    total_coauthor_count_year = 0

    start_coauthor_count_list[0] = 0
    start_coauthor_count_year_list[0] = 0

    for i in range(1, currentAcademicYearIndex + 1):
        if studentAcademicYearList[i - 1] in studentCoAuthorWeightedPaperCountMap:
            start_coauthor_count_list[i] = start_coauthor_count_list[
                i - 1
            ] + studentCoAuthorWeightedPaperCountMap[
                studentAcademicYearList[i - 1]
            ] * min(
                SUPERVISED_YEAR_MODIFIER[i - 1],
                SUPERVISED_PAPER_MODIFIER[int(start_student_count_list[i - 1])],
            )
            start_coauthor_count_year_list[i] = (
                start_coauthor_count_year_list[i - 1] + 1
            )
        else:
            start_coauthor_count_list[i] = start_coauthor_count_list[i - 1]
            start_coauthor_count_year_list[i] = start_coauthor_count_year_list[i - 1]

    end_coauthor_count_list[max_student_academic_year] = 0
    end_coauthor_count_year_list[max_student_academic_year] = 0

    for i in range(max_student_academic_year - 1, currentAcademicYearIndex - 1, -1):
        if studentAcademicYearList[i + 1] in studentCoAuthorWeightedPaperCountMap:
            end_coauthor_count_list[i] = end_coauthor_count_list[
                i + 1
            ] + studentCoAuthorWeightedPaperCountMap[
                studentAcademicYearList[i + 1]
            ] * min(
                SUPERVISED_YEAR_MODIFIER[i + 1],
                SUPERVISED_PAPER_MODIFIER[int(start_student_count_list[i + 1])],
            )
            end_coauthor_count_year_list[i] = end_coauthor_count_year_list[i + 1] + 1
        else:
            end_coauthor_count_list[i] = end_coauthor_count_list[i + 1]
            end_coauthor_count_year_list[i] = end_coauthor_count_year_list[i + 1]

    total_coauthor_count = (
        start_coauthor_count_list[currentAcademicYearIndex]
        + end_coauthor_count_list[currentAcademicYearIndex]
        + studentCoAuthorWeightedPaperCountMap[
            studentAcademicYearList[currentAcademicYearIndex]
        ]
        * min(
            SUPERVISED_YEAR_MODIFIER[currentAcademicYearIndex],
            SUPERVISED_PAPER_MODIFIER[
                int(start_student_count_list[currentAcademicYearIndex])
            ],
        )
    )

    total_coauthor_count_year = (
        start_coauthor_count_year_list[currentAcademicYearIndex]
        + end_coauthor_count_year_list[currentAcademicYearIndex]
        + 1
    )

    # iterate all possible year span (window) to compute the max supervisedRate

    maxSupervisedRate = 0.0

    for start_year_index in range(0, currentAcademicYearIndex + 1):
        for end_year_index in range(
            currentAcademicYearIndex, max_student_academic_year + 1
        ):
            # there is a problem here: the co-authorship can happen in the same year,
            # because the surrounding years may not have co-authorship between student and supervisor
            # then the small window with year_span >= 2 can still be the maximal because the co-authorship
            # are too centralized in the same year
            #
            # we solve it by using a count list for co-authorship years
            #
            if (end_year_index - start_year_index + 1) < MIN_SUPERVISED_YEAR_SPAN:
                continue

            coauthor_count_year = (
                total_coauthor_count_year
                - start_coauthor_count_year_list[start_year_index]
                - end_coauthor_count_year_list[end_year_index]
            )

            if coauthor_count_year < MIN_SUPERVISED_YEAR_SPAN:
                continue

            denominator = (
                total_student_count
                - start_student_count_list[start_year_index]
                - end_student_count_list[end_year_index]
            )

            numerator = (
                total_coauthor_count
                - start_coauthor_count_list[start_year_index]
                - end_coauthor_count_list[end_year_index]
            )

            if numerator < MIN_SUPERVISED_PAPER_SPAN:
                continue

            supervisedRate = numerator / denominator

            if supervisedRate > maxSupervisedRate:
                maxSupervisedRate = supervisedRate

    P_advisee = maxSupervisedRate
    maxSupervisedRate = min(1.0, maxSupervisedRate / MIN_SUPERVISED_RATE)

    # compute supervising rate

    supervisorPaperCountMap = topAuthorPaperCountMap[supervisorID]

    # the sorted list of years that the supervisor has paper publication

    supervisorAcademicYearList = sorted(supervisorPaperCountMap.keys())
    currentAcademicYearIndex = supervisorAcademicYearList.index(year)

    total_supervisor_count = 0
    for i in range(currentAcademicYearIndex):
        total_supervisor_count = (
            total_supervisor_count
            + supervisorPaperCountMap[supervisorAcademicYearList[i]]
        )

    coAuthorID = studentID + "-" + supervisorID
    studentCoAuthorPaperCountMap = coAuthorPaperCountMap[coAuthorID]

    coAuthorAcademicYearList = sorted(studentCoAuthorPaperCountMap.keys())
    currentAcademicYearIndex = coAuthorAcademicYearList.index(year)

    total_coauthor_count = 0
    for i in range(currentAcademicYearIndex):
        total_coauthor_count = (
            total_coauthor_count
            + studentCoAuthorPaperCountMap[coAuthorAcademicYearList[i]]
        )
    # if(total_coauthor_count==0):
    #     print(currentAcademicYearIndex)
    #     print(supervisorAcademicYearList)
    #     print(supervisorPaperCountMap)
    #     print(coAuthorAcademicYearList)
    #     print(studentCoAuthorPaperCountMap)
    supervisingRate = 0.0

    denominator = total_coauthor_count
    numerator = total_supervisor_count - total_coauthor_count

    if numerator < 0:
        print(
            "Error in computation, supervisor paper count smaller than co-author paper count:",
            studentID,
            supervisorID,
        )
        supervisingRate = 0.0
    elif numerator == 0:
        supervisingRate = 0.0
    elif denominator == 0:
        supervisingRate = numerator
    else:
        supervisingRate = numerator / denominator

    P_advisor = supervisingRate
    supervisingRate = min(1.0, supervisingRate / MIN_SUPERVISING_RATE)
    return maxSupervisedRate * supervisingRate, P_advisor, P_advisee


# settings
# reload(sys)
# sys.setdefaultencoding("utf-8")

host = "127.0.0.1"
port = "3306"
database = "scigene_acl_anthology_hIpcg"
usr = "root"
pwd = "Vis_2014"


ARC_AUTHOR = "$$ARCAUTHOR$$"
NUM_TOP_AUTHORS = "$$NUMTOPAUTHORS$$"

MIN_STUDENT_AUTHOR_ORDER_STRING = "$$MINSTUDENTAUTHORORDER$$"

createFirstAuthorTmp = "create table firstAuthorTmp select PA2.authorID as firstAuthorID, PA1.authorID as topAuthorID from scigene_acl_anthology.authors_ARC_hI as A join scigene_acl_anthology.paper_author_ARC as PA1 on A.authorID = PA1.authorID join scigene_acl_anthology.paper_author_ARC as PA2 on PA1.paperID = PA2.paperID where authorRank <= $$NUMTOPAUTHORS$$ and PA1.authorOrder > 1 and PA2.authorOrder = 1 group by PA2.authorID, PA1.authorID;"
createFirstAuthorIndex = (
    "create index first_author_index on firstAuthorTmp(firstAuthorID);"
)
createTopAuthorIndex = "create index top_author_index on firstAuthorTmp(topAuthorID);"
selectFirstAuthorPaperCount = "select authorID, authorOrder, year, count(*) as cnt from scigene_acl_anthology.paper_author_ARC as PA, scigene_acl_anthology.papers_ARC as P where authorID in (select distinct firstAuthorID from firstAuthorTmp) and PA.paperID = P.paperID group by authorID, authorOrder, year;"
selectFirstTopCoAuthorPaperCount = "select firstAuthorID, topAuthorID, PA1.authorOrder as firstAuthorOrder, year, count(*) as cnt from firstAuthorTmp join scigene_acl_anthology.paper_author_ARC as PA1 on firstAuthorID = PA1.authorID join scigene_acl_anthology.paper_author_ARC as PA2 on topAuthorID = PA2.authorID and PA1.paperID = PA2.paperID and PA1.authorOrder <= $$MINSTUDENTAUTHORORDER$$ and PA1.authorOrder < PA2.authorOrder join scigene_acl_anthology.papers_ARC as P on PA1.paperID = P.paperID group by firstAuthorID, topAuthorID, PA1.authorOrder, year;"
selectTopAuthorPaperCount = "select A.authorID, year, count(*) as cnt from scigene_acl_anthology.authors_ARC_hI as A join scigene_acl_anthology.paper_author_ARC as PA on authorRank <= $$NUMTOPAUTHORS$$ and A.authorID = PA.authorID  join scigene_acl_anthology.papers_ARC as P on PA.paperID = P.paperID group by A.authorID, year;"

dropTmpTable = "drop table firstAuthorTmp;"

selectFirstTopCoAuthorPaperCount = selectFirstTopCoAuthorPaperCount.replace(
    MIN_STUDENT_AUTHOR_ORDER_STRING, str(MIN_STUDENT_AUTHOR_ORDER)
)

createFirstAuthorTmp = createFirstAuthorTmp.replace(
    NUM_TOP_AUTHORS, str(numOfTopAuthors)
)
selectTopAuthorPaperCount = selectTopAuthorPaperCount.replace(
    NUM_TOP_AUTHORS, str(numOfTopAuthors)
)

conn = ConnectMySQLDB(host, port, database, usr, pwd)
db_cursor = conn.cursor()

dropTmpTable = "drop table firstAuthorTmp;"
# pre-compute some maps
try:
    db_cursor.execute(dropTmpTable)
    conn.commit()
except:
    pass
db_cursor.execute(createFirstAuthorTmp)
conn.commit()
db_cursor.execute(createFirstAuthorIndex)
conn.commit()
db_cursor.execute(createTopAuthorIndex)
conn.commit()

print("Create temp table for the list of first authors!")

db_cursor.execute(selectFirstAuthorPaperCount)
rows = db_cursor.fetchall()

firstAuthorPaperCountMap = {}
firstAuthorWeightedPaperCountMap = {}

for row in rows:
    authorID = row[0].strip()
    authorOrder = int(row[1])
    year = int(row[2])
    count = int(row[3])

    yearCountMap = None
    if authorID in firstAuthorPaperCountMap:
        yearCountMap = firstAuthorPaperCountMap[authorID]
    else:
        yearCountMap = {}
        firstAuthorPaperCountMap[authorID] = yearCountMap

    if year in yearCountMap:
        yearCountMap[year] = yearCountMap[year] + count
    else:
        yearCountMap[year] = count

    if authorOrder > MIN_STUDENT_AUTHOR_ORDER:
        continue

    yearCountMap = None
    if authorID in firstAuthorWeightedPaperCountMap:
        yearCountMap = firstAuthorWeightedPaperCountMap[authorID]
    else:
        yearCountMap = {}
        firstAuthorWeightedPaperCountMap[authorID] = yearCountMap

    if year in yearCountMap:
        yearCountMap[year] = yearCountMap[year] + count / authorOrder
    else:
        yearCountMap[year] = count / authorOrder

print("Pre-compute first-author maps!")

db_cursor.execute(selectFirstTopCoAuthorPaperCount)
rows = db_cursor.fetchall()

coAuthorWeightedPaperCountMap = {}
coAuthorPaperCountMap = {}

for row in rows:
    coAuthorID = row[0].strip() + "-" + row[1].strip()
    authorOrder = int(row[2])
    year = int(row[3])
    count = int(row[4])

    yearCountMap = None
    if coAuthorID in coAuthorWeightedPaperCountMap:
        yearCountMap = coAuthorWeightedPaperCountMap[coAuthorID]
    else:
        yearCountMap = {}
        coAuthorWeightedPaperCountMap[coAuthorID] = yearCountMap

    if year in yearCountMap:
        yearCountMap[year] = yearCountMap[year] + count / authorOrder
    else:
        yearCountMap[year] = count / authorOrder

    yearCountMap = None
    if coAuthorID in coAuthorPaperCountMap:
        yearCountMap = coAuthorPaperCountMap[coAuthorID]
    else:
        yearCountMap = {}
        coAuthorPaperCountMap[coAuthorID] = yearCountMap

    if year in yearCountMap:
        yearCountMap[year] = yearCountMap[year] + count
    else:
        yearCountMap[year] = count

print("Pre-compute co-author maps!")

db_cursor.execute(selectTopAuthorPaperCount)
rows = db_cursor.fetchall()

topAuthorPaperCountMap = {}

for row in rows:
    authorID = row[0].strip()
    year = int(row[1])
    count = int(row[2])

    yearCountMap = None
    if authorID in topAuthorPaperCountMap:
        yearCountMap = topAuthorPaperCountMap[authorID]
    else:
        yearCountMap = {}
        topAuthorPaperCountMap[authorID] = yearCountMap

    if year in yearCountMap:
        yearCountMap[year] = yearCountMap[year] + count
    else:
        yearCountMap[year] = count

print("Pre-compute top author maps!")

# select all top ARC authors

selectTopARCAuthors = "select authorID, name, authorRank from scigene_acl_anthology.authors_ARC_hI where authorRank <= $$NUMTOPAUTHORS$$;"
selectTopARCAuthors = selectTopARCAuthors.replace(NUM_TOP_AUTHORS, str(numOfTopAuthors))

selectTopAuthorPapers = (
    "select paperID, year, firstAuthorID,isKeyPaper from papers_arc_$$ARCAUTHOR$$"
)
updateTopAuthorPapers = (
    "update papers_arc_$$ARCAUTHOR$$ set isKeyPaper = ? where paperID = ?"
)

db_cursor.execute(selectTopARCAuthors)
rows = db_cursor.fetchall()
P_Advisor=[]
P_Advisee=[]
IsKeyPaper=[]
# process each author
for row in rows:

    topAuthorID = str(row[0].strip())
    authorName = str(row[1].strip())
    rank = int(row[2])

    authorTableName = "".join(filter(str.isalpha, authorName)).lower() + str(rank)

    selectTopAuthorPapers_author = selectTopAuthorPapers.replace(
        ARC_AUTHOR, authorTableName
    )
    updateTopAuthorPapers_author = updateTopAuthorPapers.replace(
        ARC_AUTHOR, authorTableName
    )
    authorcsv_name = "./graph/papers_arc_"+authorTableName+".csv"
    authorcsv = pd.read_csv(authorcsv_name,header=None)
    db_cursor.execute(selectTopAuthorPapers_author)
    paper_rows = db_cursor.fetchall()

    idx = 0
    # process each paper of the author
    for paper_row in paper_rows:

        paperID = str(paper_row[0].strip())
        paperYear = int(paper_row[1])
        firstAuthorID = str(paper_row[2].strip())

        isKeyPaper = 0

        if firstAuthorID == topAuthorID:
            isKeyPaper = 1
        else:
            supervisor_rate, P_advisor, P_advisee = computeSupervisorRate(
                firstAuthorID,
                topAuthorID,
                paperYear,
                firstAuthorPaperCountMap,
                firstAuthorWeightedPaperCountMap,
                coAuthorWeightedPaperCountMap,
                coAuthorPaperCountMap,
                topAuthorPaperCountMap,
            )
            P_Advisor.append(P_advisor)
            P_Advisee.append(P_advisee)

            # if supervisor_rate >= MIN_SUPERVISOR_RATE:
            #     isKeyPaper = supervisor_rate

            isKeyPaper = supervisor_rate
            IsKeyPaper.append(isKeyPaper)
            if(abs(authorcsv.iloc[idx,6]-isKeyPaper)>=0.0001):
                print(authorcsv.iloc[idx,6],isKeyPaper,idx)
            authorcsv.iloc[idx,6] = isKeyPaper
        idx += 1
        # db_cursor.execute(updateTopAuthorPapers_author, isKeyPaper, paperID)
    authorcsv.to_csv(authorcsv_name,index=None,header=None)
    print(
        "Update key papers for ARC author ",
        authorName,
        " with rank ",
        str(rank),
        ":",
        authorTableName,
    )
    conn.commit()

# import numpy as np

# print(P_Advisee)
# n,bins,patches = plt.hist(P_Advisee,120,range=(0.0,1.2))
# P_Advisee = pd.DataFrame([n,bins])
# P_Advisee = P_Advisee.T
# print(P_Advisee)
# P_Advisee.to_csv("P_Advisee.csv",index=None,header=None)

# print(IsKeyPaper)
# n,bins,patches = plt.hist(IsKeyPaper,100)
# IsKeyPaper = pd.DataFrame([n,bins])
# IsKeyPaper = IsKeyPaper.T
# print(IsKeyPaper)
# IsKeyPaper.to_csv("IsKeyPaper.csv",index=None,header=None)

# print(P_Advisor)
# n,bins,patches = plt.hist(P_Advisor,int(max(np.array(P_Advisor))))
# P_Advisor = pd.DataFrame([n,bins])
# P_Advisor = P_Advisor.T
# print(P_Advisor)
# P_Advisor.to_csv("P_Advisor.csv",index=None,header=None)

# path = "P_Advisee.jpg"
# draw_distribution_histogram(P_Advisee, path, True, False)

db_cursor.execute(dropTmpTable)
conn.commit()
db_cursor.close()
conn.close()
