'''
Script for preparing data from scene graphs to be used at train time
'''
from data.gqaDataset import gqa
from kb_retrieval.graph_construction import GraphConstructor
from kb_retrieval.candidate_trimming import CandidateGenerator
from tqdm import tqdm
import argparse
from networkx.readwrite import gpickle
from networkx import set_edge_attributes, get_edge_attributes

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers to use for multiprocessing operations')
    parser.add_argument('--split', type=str, default='train', help="train test or validation split")
    return parser.parse_args()

def get_pair(entity):
    '''
    takes a (entity_id,node_attributes) pair. If the node attributes are empty it returns (entity_id,entity_id), otherwise
    it should extract the label
    '''
    if not entity[1]: # checks if the dictionary is empty
        return entity[0], entity[0]
    else:
        return entity[0], entity[1]['label']

if __name__ == '__main__':
    args = parse_arguments()
    ds = gqa(args.split)
    for i,(question,answer,graph) in tqdm(enumerate(ds)):
        all_entities= graph.nodes.items()
        all_entities = {get_pair(entity)[1]:get_pair(entity)[0] for entity in all_entities}
        all_relations = set(get_edge_attributes(graph,"label").values())
        all_relations = {relation:relation for relation in all_relations}  
        gc = GraphConstructor()
        # triples, background_entities = query_wikidata(de,question,conv_context,background)
        cg = CandidateGenerator(5,0.8)
        question_entities= cg.trim(question,all_entities)
        background_entities = cg.trim(answer,all_entities)
        background_relation_labels = cg.trim(answer,all_relations)
        gc.input_nx_graph_with_trimming(graph, question_entities, 3)
        potentially_relevant = gc.extract_relevant_nodes(graph,background_entities,1)
        edge_label_dict = {}
        processed_graph = gc.build_graph()
        edges = processed_graph.edges(data=True)
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
    
        set_edge_attributes(processed_graph,edge_label_dict,'relevance_label')
        gpickle.write_gpickle(processed_graph,f"datasets/gqa/graph{i}.json")

