rm -f *.txt
cd graph 
rm -f *.csv
cd ..
python3 downloadGraph.py $1
python3 find_recall_precision.py $1 $2

# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 1 1
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 1 1.5
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 1 2
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 1 2.5
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 1 3
# python find_recall_precision.py $1 $2

# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 2 1
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 2 1.5
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 2 2
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 2 2.5
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 2 3
# python find_recall_precision.py $1 $2

# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 3 1
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 3 1.5
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 3 2
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 3 2.5
# python find_recall_precision.py $1 $2
# rm -f *.txt
# python Descriptive_advisor-advisee_detection_method.py 200 3 3
# python find_recall_precision.py $1 $2

