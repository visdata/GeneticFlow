rm ./data/csv_unprocessed/*.csv
rm ./data/csv/*.csv
rm ./data/*.csv
cp ../../../GeneticFlow_Signatures/GNN_NLP_data/*.csv ./data/
cp ../../../../Bibliographic_Networks/$1/BiblioCouplingNetwork/*.csv ./data/csv_unprocessed/
rm data/processed/*

cp ../../../GeneticFlow_Signatures/GNN_NLP_data/csv/top_field_authors.csv ./data/csv_unprocessed/
cd data/csv_unprocessed

python data_process.py

cd ../..

