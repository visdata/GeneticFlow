rm ./data/csv_unprocessed/*.csv
rm ./data/csv/*.csv
cp ../../../../Bibliographic_Networks/$1/coauthorNetwork/*.csv ./data/csv_unprocessed/
rm data/processed/*
cp ../../../GeneticFlow_Signatures/GNN_othersField/data/csv/top_field_authors.csv ./data/csv_unprocessed/
cd data/csv_unprocessed

python data_process.py

cd ../..

python main.py