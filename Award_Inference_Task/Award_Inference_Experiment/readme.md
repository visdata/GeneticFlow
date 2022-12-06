# Code Flow

## GeneticFlow Signatures

    For NLP, using GeneticFlow, we perform 1 and 2 to process the graph data, 3 and 4 to predict the awardee with several of our models:
        1.cd GeneticFlow_Signatures/GNN_NLP_data
        2.sh run.sh
        3.cd ../GNN_NLP
        4.sh run.sh
        5.sh coregraph_cutedge.sh ${extend-proba}
        6.sh fullgraph_cutedge.sh  ${extend-proba}

    For OthersField, using GeneticFlow, we perform 1 and 2 to process the graph data and predict the awardee with several of our models:
        1.cd GeneticFlow_Signatures/GNN_othersField
        2.sh allfield.sh

    For pooling, ginConv, chebConv, stack experiemnts：
        1.refer to model_stack.py, models_cheb.py, models_gin.py, models_pool.py

## Bibliographic_Networks

    For NLP, we use the BiblioCouplingNetwork as an example to show how we predict the awardee with several of our models, using Bibliographic_Networks:
        1.cd GeneticFlow_Signatures/GNN_NLP_data
        2.sh run.sh
        1.cd Alternative_Method/GNN_Bibliographic_Networks/GNN_BiblioCouplingNetwork
        2.sh run_arc.sh NLP_ARC_hI
        3.python main.py --lr 0.001 --nhid 256 --dropout_ratio 0.2 --num_layers 2 --average True

    For OthersField, we take Bibliographic_Networks experiments in the field of HCI as an example, using coauthorNetwork:
        1.cd Alternative_Method/GNN_Bibliographic_Networks/GNN_coauthorNetwork
        2.sh run.sh HCI_hI
        3.python main.py

    Refer to the document below to see the optimal parameters of Bibliographic_Networks experiments in all fields:
        1.Alternative_Method/GNN_Bibliographic_Networks/params.txt

## Author-Level_Impact_Indicators

    For NLP, we perform 1 and 2 to predict the awardee with several of our models, using Author-Level_Impact_Indicators:
        1.cd Alternative_Method/Author-Level_Impact_Indicators/NLP
        2.sh run.sh
    For OthersField, we take the HCI field as an example to predict the awardee with several of our models, using Author-Level_Impact_Indicators:
        1.cd Alternative_Method/Author-Level_Impact_Indicators/othersField
        2.sh run.sh HCI_hI
