mv Non_student_* non_student/
mv *.txt student/
cd non_student
python change_filename.py
cd ..
cd student
python change_filename.py
cd ..