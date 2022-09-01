python non_graph.py

rm data/processed/*
python main.py --PATH $1 --conv_name GCNConv 
rm data/processed/*
python main.py --PATH $1 --conv_name ChebConv
rm data/processed/*
python main.py --PATH $1 --conv_name SAGEConv
rm data/processed/*
python main.py --PATH $1 --conv_name GraphConv
rm data/processed/*
python main.py --PATH $1 --conv_name GATConv
rm data/processed/*
python main.py --PATH $1 --conv_name TAGConv
rm data/processed/*
python main.py --PATH $1 --conv_name ARMAConv
rm data/processed/*
python main.py --PATH $1 --conv_name SGConv
rm data/processed/*
python main.py --PATH $1 --conv_name MFConv
rm data/processed/*
python main.py --PATH $1 --conv_name LEconv
rm data/processed/*
python main.py --PATH $1 --conv_name GINConv

# sh pool.sh $1 GCNConv
# sh pool.sh $1 ChebConv
# sh pool.sh $1 SAGEConv
# sh pool.sh $1 GraphConv
# sh pool.sh $1 GATConv
# sh pool.sh $1 TAGConv
# sh pool.sh $1 ARMAConv
# sh pool.sh $1 SGConv
# sh pool.sh $1 MFConv
# sh pool.sh $1 LEconv
# sh pool.sh $1 GINConv

# rm data/processed/*
# python main.py --conv_name ARMAConv --dropout_ratio 0.5
# rm data/processed/*
# python main.py --conv_name ARMAConv --dropout_ratio 0.2
# rm data/processed/*
# python main.py --conv_name ARMAConv --dropout_ratio 0.1
# rm data/processed/*
# python main.py --conv_name TAGConv --dropout_ratio 0.5
# rm data/processed/*
# python main.py --conv_name TAGConv --dropout_ratio 0.3
# rm data/processed/*
# python main.py --conv_name TAGConv --dropout_ratio 0.1