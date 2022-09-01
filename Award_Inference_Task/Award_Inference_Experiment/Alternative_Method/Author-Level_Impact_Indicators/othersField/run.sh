cd ../../../GeneticFlow_Signatures/GNN_othersField
python non_graph.py
cd ../../Alternative_Method/Author-Level_Impact_Indicators/othersField
cp ../../../GeneticFlow_Signatures/GNN_othersField/non_graph.csv .
python svm.py
python xgb.py
python mlp3.py
python rf.py
