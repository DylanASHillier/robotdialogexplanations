'''
Script for preparing data to be used at train time
'''
from kb_retrieval.data_extraction import DataExtractor
from kb_retrieval.graph_construction import GraphConstructor
from kb_retrieval.candidate_trimming import CandidateGenerator
from tqdm import tqdm
from data import qtext
import argparse
from networkx.readwrite import gpickle
from networkx import set_edge_attributes, get_edge_attributes
import json
from itertools import groupby
from math import floor
import time

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='', help='dataset for training, choose from \'ropes\',\'coqa\',\'squad\'')
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers to use for multiprocessing operations')
    parser.add_argument('--split', type=str, default='train', help="train test or validation split")
    return parser.parse_args()

def batch(items,batchsize):
    l = len(items)
    items = enumerate(items)
    items = [list(g) for k,g in groupby(items,lambda x: (floor(x[0]/batchsize)))]
    return [[x[1] for x in g] for g in items]


end = 60
start =0
def query_wikidata(de, question, conv_context, background):
    entities = de.get_entities_from_falcon2(conv_context)+de.get_entities_from_falcon2(question)
    background_entities = [item['id'] for item in de.get_entities_from_falcon2(background)]
    batched_entities = batch(entities,29)
    triples = []
    for entities in tqdm(batched_entities):
        if int(end - start) <60:
            time.sleep(60-(int(end-start)))##avoid timeouts
        start = time.time()
        for triple in de.triples_from_query(entities):
            triples.append(triple)
        end = time.time()
        print(end-start)
    return triples, background_entities

def get_pair(entity):
    '''
    takes a (entity_id,node_attributes) pair. If the node attributes are empty it returns (entity_id,entity_id), otherwise
    it should extract the label
    '''
    if not entity[1]: # checks if the dictionary is empty
        return entity[0],entity[0]
    else:
        raise ValueError(f"unimplemented for {entity}")

if __name__ == '__main__':
    de = DataExtractor()
    args = parse_arguments()
    ds = None
    if args.dataset=='ropes':
        ds = qtext.QtextRopes(args.split)
    elif args.dataset=='squad':
        ds = qtext.QtextSQUAD(args.split)
    elif args.dataset=='coqa':
        ds = qtext.QtextCoQA(args.split)
    else:
        raise ValueError("dataset arg required")

    base_graph = gpickle.read_gpickle("datasets/KGs/conceptnet.json")
    all_entities= base_graph.nodes.items()
    all_entities = {get_pair(entity)[1]:get_pair(entity)[0] for entity in all_entities}
    all_relations = set(get_edge_attributes(base_graph,"label").values())
    all_relations = {relation:relation for relation in all_relations}
    for i,(question,conv_context,background) in tqdm(enumerate(ds)):    
        gc = GraphConstructor()
        # triples, background_entities = query_wikidata(de,question,conv_context,background)
        cg = CandidateGenerator(5,0.8)
        question_entities= cg.trim(question,all_entities)
        background_entities = cg.trim(background,all_entities)
        conv_context_entities = cg.trim(conv_context,all_entities)
        background_relation_labels = cg.trim(background,all_relations)
        gc.input_nx_graph_with_trimming(base_graph, question_entities+conv_context_entities, 1)
        graph = gc.build_graph()
        potentially_relevant = gc.extract_relevant_nodes(base_graph,background_entities,1)
        edge_label_dict = {}
        edges = graph.edges(data=True)
        for edge in edges:
            u,v,label = edge
            score = 0
            if u in potentially_relevant:
                score+=1
            if u in background_entities:
                score+=1
            if v in potentially_relevant:
                score+=1
            if v in background_entities:
                score+=1
            if label["label"] in background_relation_labels:
                score+=1
            edge_label_dict[(u,v)]=score
    
        set_edge_attributes(graph,edge_label_dict,'relevance_label')
        # with open(f"datasets/{args.dataset}/graph{i}.json",'w') as fp:
        gpickle.write_gpickle(graph,f"datasets/{args.dataset}/graph{i}.json")
    
