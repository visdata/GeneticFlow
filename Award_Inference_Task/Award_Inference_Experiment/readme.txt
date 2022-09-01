Code Flow:
GeneticFlow Signatures:
    For NLP, using GeneticFlow, we perform 1 and 2 to process the graph data, 3 and 4 to predict the awardee with several of our models:
        1.cd GeneticFlow_Signatures/GNN_NLP_data
        2.sh run.sh
        3.cd ../GNN_NLP
        4.sh dataset.sh

    For OthersField, using GeneticFlow, we perform 1 and 2 to process the graph data and predict the awardee with several of our models:
        1.cd GeneticFlow_Signatures/GNN_othersField
        2.sh allfield.sh

Bibliographic_Networks:
    For all field, we perform 1 and 2 to predict the awardee with several of our models, using Bibliographic_Networks:
        1.cd GeneticFlow_Signatures/GNN_othersField
        2.sh allfieldBibliographicGraphs.sh

Author-Level_Impact_Indicators:
    For NLP, we perform 1 and 2 to predict the awardee with several of our models, using Author-Level_Impact_Indicators:
        1.cd Alternative_Method/Author-Level_Impact_Indicators/NLP
        2.sh run.sh
    For OthersField, we perform 1 and 2 to predict the awardee with several of our models, using Author-Level_Impact_Indicators:
        1.cd Alternative_Method/Author-Level_Impact_Indicators/othersField
        2.sh run.sh
