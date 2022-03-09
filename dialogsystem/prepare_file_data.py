'''
Script for preparing data to be used at train time
'''
from kb_retrieval.data_extraction import DataExtractor
from kb_retrieval.graph_construction import GraphConstructor
from tqdm import tqdm
from data import qtext
import argparse
from networkx.readwrite import json_graph
import json
from itertools import groupby
from math import floor

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='', help='dataset for training, choose from \'ropes\',\'coqa\',\'squad\'')
    return parser.parse_args()

def batch(items,batchsize):
    l = len(items)
    items = enumerate(items)
    items = [list(g) for k,g in groupby(items,lambda x: (floor(x[0]/batchsize)))]
    return [[x[1] for x in g] for g in items]


if __name__ == '__main__':
    de = DataExtractor()
    args = parse_arguments()
    ds = None
    if args.dataset=='ropes':
        ds = qtext.QtextRopes("train")
    elif args.dataset=='squad':
        ds = qtext.QtextSQUAD("train")
    elif args.dataset=='coqa':
        ds = qtext.QtextCoQA("train")
    else:
        ValueError("datset arg required")
    for i,(question,conv_context,background) in tqdm(enumerate(ds)):
        gc = GraphConstructor()
        entities = de.get_entities_from_falcon2(conv_context)+de.get_entities_from_falcon2(question)
        batched_entities = batch(entities,5)
        triples = []
        for entities in batched_entities:
            for triple in de.triples_from_query(entities):
                triples.append(triple)
        for triple in triples:
            gc.add_from_triple(triple)
        graph = gc.build_graph()
        json_g = json_graph(graph)
        json.dump(json_g,fp=f"datasets/KGs/{args.dataset}/graph{i}.json")

        
