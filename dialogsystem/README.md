# Script Guide
## Dataset Generation
Assumes you have run setup.py

These scripts are merely for extracting the graphs. The graphs themselves must still be embedded with the text. This is done using the 
### QText Datasets
1. setupconceptnet.py -> stores a json graph of the conceptnet graph
1. qtext_data_prep.py -> sets up the qtext datasets storing them as networkx graphs in json format
### GQA Dataset
1. scene_graph_data_prep.py
## Training
1. train.py which is used for training a GNN and an autoencoder used for the graph-text encoding step
1. triples_to_text_pipeline.py which is used for training the triples to text system
1. conv_qa_pipeline.py which is used for training the conv_qa system

## Evaluation
1. conv_qa_eval.py
1. gqa_eval.py - evaluates trained GNNs on the GQA dataset
1. system_eval.py - does an ablation study on the system as a whole

## Visualisation
1. visualise_graph.py
