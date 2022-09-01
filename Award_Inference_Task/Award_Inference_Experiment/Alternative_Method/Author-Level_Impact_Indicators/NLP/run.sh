cd ../../../GeneticFlow_Signatures/GNN_NLP
python non_graph.py
cd ../../Alternative_Method/Author-Level_Impact_Indicators/NLP
cp ../../../GeneticFlow_Signatures/GNN_NLP/non_graph.csv .
python svm.py
python xgb.py
python mlp.py
python rf.py
