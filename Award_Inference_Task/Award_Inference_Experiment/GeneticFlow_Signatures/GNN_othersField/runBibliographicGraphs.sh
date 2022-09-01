rm -f ./data/csv_unprocessed/*.csv
rm -f ./data/csv/*.csv
rm -f ./data/data/*.csv
rm -f ./data/data_origin/*.csv
rm -f ./data/data_with_feature/*.csv
rm -f ./data/data_origin_cut_edge/*.csv
rm -f ./data/data_with_proba/*.csv
cp ../../../GeneticFlow_Graphs/$1/*.csv ./data/csv_unprocessed/

cd data/csv_unprocessed/
python data_process.py
cd ../..

cd ../../Alternative_Method/GNN_Bibliographic_Networks/GNN_BiblioCouplingNetwork
sh four_graph.sh $1