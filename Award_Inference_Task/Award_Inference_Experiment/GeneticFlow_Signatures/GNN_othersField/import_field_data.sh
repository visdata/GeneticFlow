rm -f ./data/csv_unprocessed/*.csv
rm -f ./data/csv/*.csv
rm -f ./data/data/*.csv
rm -f ./data/data_origin/*.csv
rm -f ./data/data_with_feature/*.csv
rm -f ./data/data_origin_cut_edge/*.csv
rm -f ./data/data_with_proba/*.csv
rm -f ./data/processed/*
cp ../../../GeneticFlow_Graphs/$1/*.csv ./data/csv_unprocessed/

cd data/csv_unprocessed/
python data_process.py
cd ../..

cd data/csv/

python data-l-o.py
python data-l.py
python data-p-o.py
python data-p.py

cd ../..

cd data
cp ./csv_unprocessed/all_features.csv ./data_origin/
cp ./csv_unprocessed/all_features.csv ./data/

cp ./data/*.csv ./data_with_feature/

cd data
python add_feature.py $1
cd ..
cd data_origin
python add_feature.py $1
python data-proba.py
cd ..

cp ./data_with_feature/*.csv ./data_with_proba/

cd data_with_proba

python data-proba.py

cd ../..

rm -f ./data/processed/*

python non_graph.py