rm ./data/processed/*
python main.py --PATH ../GNN_NLP_data/data_origin --lr 0.003 --nhid 256 --dropout_ratio 0.2 --conv_name ARMAConv --num_layers 2
rm ./data/processed/*
python main.py --PATH ../GNN_NLP_data/data --lr 0.003 --nhid 256 --dropout_ratio 0.2 --conv_name ARMAConv --num_layers 1