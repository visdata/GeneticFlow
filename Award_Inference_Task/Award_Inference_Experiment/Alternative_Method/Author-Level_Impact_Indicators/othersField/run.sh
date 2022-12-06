cd ../../../GeneticFlow_Signatures/GNN_othersField
rm data/processed/*
rm non_graph.csv
sh import_field_data.sh $1
python non_graph.py
cd ../../Alternative_Method/Author-Level_Impact_Indicators/othersField
cp ../../../GeneticFlow_Signatures/GNN_othersField/non_graph.csv .
python svm.py
python xgb.py
python mlp3.py
python rf.py
