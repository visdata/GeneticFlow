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

cd data/csv/

python data-l-o.py
python data-l.py
python data-p-o.py
python data-p.py

cd ../..

cd data
cp ./csv_unprocessed/all_features.csv ./data_origin/
cp ./csv_unprocessed/all_features.csv ./data/

cp ./data/*.csv ./data_with_feature/

cd data
python add_feature.py
cd ..
cd data_origin
python add_feature.py
python data-proba.py
cd ..

cp ./data_with_feature/*.csv ./data_with_proba/

cd data_with_proba

python data-proba.py

cd ../..

rm -f ./data/processed/*

python non_graph.py

# if [[ $1 == "Database" ]];
# then    
#     sh edge.sh 0.3312097619914893
#     sh edge.sh 0.2945628736398316
#     sh edge.sh 0.2728347497305224
#     sh edge.sh 0.2562499760417522
#     sh edge.sh 0.243001036851756
#     sh edge.sh 0.2321151508680912
#     sh edge.sh 0.2214674932406847
#     sh edge.sh 0.2108957247157019
#     sh edge.sh 0.1968119632068985
#     sh edge.sh 0.1326763671826105
# fi

# if [[ $1 == "ComputerSecurity" ]];
# then    
#     sh edge.sh 0.3280083132407459
#     sh edge.sh 0.2907859260623435
#     sh edge.sh 0.2693603963732041
#     sh edge.sh 0.2535562156997571
#     sh edge.sh 0.241023798158495
#     sh edge.sh 0.2305220812637938
#     sh edge.sh 0.2208823993708347
#     sh edge.sh 0.2111383537181489
#     sh edge.sh 0.1977802337818077
#     sh edge.sh 0.1251179525879092
# fi

# if [[ $1 == "DataMining" ]];
# then    
#     sh edge.sh 0.3299216707389267
#     sh edge.sh 0.2930978339322463
#     sh edge.sh 0.2706282464293061
#     sh edge.sh 0.2542768185357805
#     sh edge.sh 0.2416391063936177
#     sh edge.sh 0.230506748667406
#     sh edge.sh 0.2203532654355266
#     sh edge.sh 0.2099498179715516
#     sh edge.sh 0.1955498846765981
#     sh edge.sh 0.1198293068754296
# fi

# if [[ $1 == "HCI" ]];
# then    
#     sh edge.sh 0.3012929884452524
#     sh edge.sh 0.2705249054661515
#     sh edge.sh 0.2540440118623056
#     sh edge.sh 0.2417982723661482
#     sh edge.sh 0.2318827689895561
#     sh edge.sh 0.2231889323583465
#     sh edge.sh 0.2149492917755229
#     sh edge.sh 0.2063490226308
#     sh edge.sh 0.1929942442287588
#     sh edge.sh 0.1202869921815949
# fi

# if [[ $1 == "ProgrammingLanguage" ]];
# then    
#     sh edge.sh 0.3131213684173997
#     sh edge.sh 0.2802514352497308
#     sh edge.sh 0.2612560412284764
#     sh edge.sh 0.2479110390519304
#     sh edge.sh 0.2369309162031694
#     sh edge.sh 0.2272066393671901
#     sh edge.sh 0.2179794344207746
#     sh edge.sh 0.2086003296107567
#     sh edge.sh 0.1954744698725847
#     sh edge.sh 0.1309221399816956
# fi

python main.py --PATH ./data/data_origin --conv_name ChebConv
rm -f ./data/processed/*
python main.py --PATH ./data/data_origin --conv_name ARMAConv
# rm -f ./data/processed/*
# python main.py --PATH ./data/data --conv_name ChebConv
# rm -f ./data/processed/*
# python main.py --PATH ./data/data_with_feature --conv_name ChebConv
# rm -f ./data/processed/*
# python main.py --PATH ./data/data_with_proba --conv_name ChebConv
# rm -f ./data/processed/*
# python main.py --PATH ./data/data_with_proba --conv_name ARMAConv

# rm -f ./data/processed/*
# python main.py --PATH ./data/data_origin --conv_name SAGEConv
# rm -f ./data/processed/*
# python main.py --PATH ./data/data --conv_name SAGEConv

# rm -f ./data/processed/*
# python main.py --PATH ./data/data_origin --conv_name TAGConv
# rm -f ./data/processed/*
# python main.py --PATH ./data/data --conv_name TAGConv
# rm -f ./data/processed/*
# python main.py --PATH ./data/data_with_feature --conv_name TAGConv
# rm -f ./data/processed/*
# python main.py --PATH ./data/data_with_proba --conv_name TAGConv


# *database:
# Non-fellow 1 : Wolfgang Lehner papers_num: 237 order: 4
# Non-fellow 150 : Christopher J. Kane papers_num: 70 order: 216
# fellow 1 : Elisa Bertino papers_num: 274 order: 1
# fellow 50 : Juliana Freire papers_num: 98 order: 101

# ComputerGraphicsImages:
# Non-fellow 1 : Hans Peter Seidel papers_num: 399 order: 1
# Non-fellow 150 : Ronald S. Cok papers_num: 64 order: 176
# fellow 1 : Markus Gross papers_num: 223 order: 5
# fellow 50 : Donna Cox papers_num: 28 order: 1147

# *ComputerSecurity:
# Non-fellow 1 : Willy Susilo papers_num: 410 order: 2
# Non-fellow 150 : Atsuko Miyaji papers_num: 106 order: 178
# fellow 1 : Elisa Bertino papers_num: 458 order: 1
# fellow 50 : Geoffrey M. Voelker papers_num: 61 order: 620

# *DataMining:
# Non-fellow 1 : Witold Pedrycz papers_num: 516 order: 3
# Non-fellow 150 : Aristides Gionis papers_num: 126 order: 191
# fellow 1 : Philip S. Yu papers_num: 840 order: 1
# fellow 50 : Lise Getoor papers_num: 115 order: 238

# *HCI:
# Non-fellow 1 : Jean Vanderdonckt papers_num: 330 order: 2
# Non-fellow 150 : Regan L. Mandryk papers_num: 89 order: 205
# fellow 1 : Mark Billinghurst papers_num: 354 order: 1
# fellow 50 : Abigail Sellen papers_num: 96 order: 158

# *ProgrammingLanguage:
# Non-fellow 1 : JosÃ© Meseguer papers_num: 242 order: 2
# Non-fellow 150 : GermÃ¡n Puebla papers_num: 78 order: 197
# fellow 1 : Simon Jones papers_num: 247 order: 1
# fellow 50 : Barbara G. Ryder papers_num: 76 order: 208

# *TheoreticalComputerScience:
# Non-fellow 1 : Kim Guldstrand Larsen papers_num: 266 order: 2
# Non-fellow 150 : Gianluigi Zavattaro papers_num: 93 order: 189
# fellow 1 : Moshe Y. Vardi papers_num: 312 order: 1
# fellow 50 : Karem A. Sakallah papers_num: 80 order: 296

# SoftwareEngineering:
# Non-fellow 1 : Computer Staff papers_num: 294 order: 1
# Non-fellow 90 : Yann papers_num: 82 order: 104
# fellow 1 : Barry Boehm papers_num: 234 order: 2
# fellow 30 : David Harel papers_num: 46 order: 445

# SpeechRecognition:
# Non-fellow 1 : Shrikanth S. Narayanan papers_num: 613 order: 1
# Non-fellow 90 : Tetsuya Takiguchi papers_num: 164 order: 98
# fellow 1 : Hermann Ney papers_num: 549 order: 3
# fellow 30 : Gunnar Fant papers_num: 53 order: 987
