rm -f ./csv_unprocessed/*.csv
rm -f ./csv/*.csv
rm -f ./data/*.csv
rm -f ./data_origin/*.csv
rm -f ./data_with_feature/*.csv
rm -f ./data_with_proba/*.csv
rm -f ./data_cut_edge/*.csv
rm -f ./data_origin_with_feature/*.csv
rm -f ./data_origin_cut_edge/*.csv
rm -f ./data_origin_non_cut_edge/*.csv

cp ../../../GeneticFlow_Graphs/NLP_ARC_hI/*.csv ./csv_unprocessed/

cd ./csv_unprocessed/
python data_process.py
cd ..

cd ./csv/
python data-l-o.py
python data-l.py
python data-p-o.py
python data-p.py
cd ..


cp ./csv_unprocessed/all_features.csv ./data/
cp ./csv_unprocessed/all_features.csv ./data_origin/
cp ./data/*.csv ./data_with_feature/
cp ./data_origin/*.csv ./data_origin_with_feature/

#add_feature.py convert files in data (or data_origin), sent out put to data_with_feature (or data_origin_with_feature)
cd data
python add_feature.py
cd ..

cd data_origin
python add_feature.py
cd ..

cp ./data_with_feature/*.csv ./data_with_proba/
cd data_with_proba
python data-proba.py
cd ..

