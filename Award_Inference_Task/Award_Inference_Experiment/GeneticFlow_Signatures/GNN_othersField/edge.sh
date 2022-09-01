cd data
rm -f ./data_origin_cut_edge/*.csv
cp ./data/*.csv ./data_origin_cut_edge/
cd data_origin_cut_edge
python cut_arc.py $1
cd ../..
rm -f ./data/processed/*
python main.py --PATH ./data/data_origin_cut_edge --conv_name ARMAConv