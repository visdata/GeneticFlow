# rm data/processed/*
# python main.py --PATH $1 --conv_name $2 --pool_name SAGPooling
# rm data/processed/*
# python main.py --PATH $1 --conv_name $2 --pool_name TopKPooling
# rm data/processed/*
# python main.py --PATH $1 --conv_name $2 --pool_name ASAPooling
rm data/processed/*
python main.py --PATH $1 --conv_name $2 --pool_name EdgePooling
# rm data/processed/*
# python main.py --PATH $1 --conv_name $2 --pool_name HGPSLPool