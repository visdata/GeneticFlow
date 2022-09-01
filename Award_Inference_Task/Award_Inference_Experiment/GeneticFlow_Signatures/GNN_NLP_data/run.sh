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

cp ../../../GeneticFlow_Graphs/NLP/*.csv ./csv_unprocessed/

cd ./csv_unprocessed/
python data_process.py
cd ..

cd ./csv/

python data-l-o.py
python data-l.py
# python data-n-l.py
python data-p-o.py
python data-p.py
# python data-n-p.py

cd ..

cp ./csv_unprocessed/all_features.csv ./data/
cp ./csv_unprocessed/all_features.csv ./data_origin/
cp ./data/*.csv ./data_with_feature/
cp ./data_origin/*.csv ./data_origin_with_feature/

# cd data
# python add_feature.py
# cd ..

cd data_origin
python add_feature.py
cd ..

# cp ./data_with_feature/*.csv ./data_with_proba/
# cd data_with_proba
# python data-proba.py
# cd ..

cp ./data_with_feature/*.csv ./data_cut_edge/
cp ./data_with_feature/*.csv ./data_origin_cut_edge/
cp ./data_with_feature/*.csv ./data_origin_non_cut_edge/

# cd data_cut_edge
# python cut_arc.py
# cd ..

rm -f ./data_origin_cut_edge/*.csv
cp ./data_with_feature/*.csv ./data_origin_cut_edge/
cd data_origin_cut_edge
python cut_arc.py 0.372
cd ../GNN_NLP
sh run.sh
cd ../GNN_NLP_data

# cd data_origin_non_cut_edge
# python cut_arc.py
# cd ..
