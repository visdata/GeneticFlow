cd ../GNN_NLP_data/
rm -f ./data_origin_cut_edge/*.csv
cp ./data_origin_with_feature/*.csv ./data_origin_cut_edge/
cd data_origin_cut_edge
python cut_arc.py $1
cd ../../GNN_NLP
rm -f ./data/processed/*
python main.py --PATH ../GNN_NLP_data/data_origin_cut_edge --lr 0.003 --nhid 256 --dropout_ratio 0.2 --conv_name ARMAConv --num_layers 2
