sh import_field_data.sh Database_hI
rm -f ./data/processed/*
python main.py --PATH ./data/data_origin --lr 0.001 --nhid 128 --dropout_ratio 0.3 --conv_name ARMAConv --num_layers 2 --average True
rm -f ./data/processed/*
python main.py --PATH ./data/data --lr 0.0005 --nhid 256 --dropout_ratio 0.0 --conv_name ARMAConv --num_layers 1 --average True

sh import_field_data.sh ComputerSecurity_hI
rm -f ./data/processed/*
python main.py --PATH ./data/data_origin --lr 0.002 --nhid 256 --dropout_ratio 0.5 --conv_name ARMAConv --num_layers 1 --average True
rm -f ./data/processed/*
python main.py --PATH ./data/data --lr 0.003 --nhid 128 --dropout_ratio 0.3 --conv_name ARMAConv --num_layers 1 --average True

sh import_field_data.sh DataMining_hI
rm -f ./data/processed/*
python main.py --PATH ./data/data_origin --lr 0.004 --nhid 256 --dropout_ratio 0.1 --conv_name ARMAConv --num_layers 2 --average True
rm -f ./data/processed/*
python main.py --PATH ./data/data --lr 0.003 --nhid 256 --dropout_ratio 0.3 --conv_name ARMAConv --num_layers 1 --average True

sh import_field_data.sh HCI_hI
rm -f ./data/processed/*
python main.py --PATH ./data/data_origin --lr 0.003 --nhid 256 --dropout_ratio 0.4 --conv_name ARMAConv --num_layers 1 --average True
rm -f ./data/processed/*
python main.py --PATH ./data/data --lr 0.001 --nhid 256 --dropout_ratio 0.1 --conv_name ARMAConv --num_layers 2 --average True

sh import_field_data.sh SoftwareEngineering_hI
rm -f ./data/processed/*
python main.py --PATH ./data/data_origin --lr 0.001 --nhid 256 --dropout_ratio 0.5 --conv_name ARMAConv --num_layers 1 --average True
rm -f ./data/processed/*
python main.py --PATH ./data/data --lr 0.001 --nhid 256 --dropout_ratio 0.1 --conv_name ARMAConv --num_layers 1 --average True
