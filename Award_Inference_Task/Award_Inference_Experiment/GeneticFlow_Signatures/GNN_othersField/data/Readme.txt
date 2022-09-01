./csv_unprocessed:
    Initial graph data.
./csv:
    The graph data after author-sampling. The sample base guarantees that at least 50 award recipients and 150 other authors are included, which are obtained through two separate uniform sampling.
./data_origin:
    The full GF graphs.(Removing non-self-citing edges of the graph)
./data:
    The core GF graphs.
./data_origin_cut_edge:
    The full GF graphs with core-extend edges.(after cutting the non-core-extend edges)
./data_with_feature:
    The core GF graphs with edge features.
./data_with_feature_proba:
    The core GF graphs with edge-extend probability.
./fellow.csv:
    Award authors.(field rank)
./non_fellow.csv:
    Non award authors.(field rank)


