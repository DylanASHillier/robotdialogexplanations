from data.gqaDataset import gqa
from models.GraphEmbedder import GraphTransformer
from models.kgqueryextract import LightningKGQueryMPNN
from kb_retrieval.graph_construction import GraphConstructor
from kb_retrieval.candidate_trimming import CandidateGenerator
from tqdm import tqdm
from networkx import set_edge_attributes, get_edge_attributes
from numpy import random
from torch import mean, topk, nan_to_num

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
    ds = gqa("val")
    random.seed(123)
    idxs = random.choice(range(len(ds)), size=1000)
    mpnn = LightningKGQueryMPNN.load_from_checkpoint("dialogsystem/trained_models/gqanew.ckpt")
    mpnn.avg_pooling=False
    mpnn.k = 10
    graph_transformer = GraphTransformer(lm_string="t5-small")
    samples= {"failed_extraction":0, "precision@1":0, "precision@3":0, "precision@10":0, "recall@1":0, "recall@3":0, "recall@10":0}
    for i in tqdm(idxs):
        question, answer, graph = ds[i]
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
        if processed_graph.size()==0:
            samples["failed_extraction"]+=1
        else:
            processed_graph = graph_transformer(processed_graph, relevance_label=True)
            data = graph_transformer.add_query(processed_graph,question)

            output = mpnn(data.x, data.edge_index)
            for i in [1,3,10]:
                topkvals,topkindices = topk(output,min(i,output.shape[0]))
                labels = data.y[topkindices]
                topkvals = topkvals.tolist()
                topkindices = topkindices.tolist()
                if sum(data.y)==0:
                    samples["failed_extraction"]+=1
                else:
                    precision = sum(labels>2)/len(labels)
                    recall = nan_to_num(sum(labels>2)/sum(data.y>2),0)
                    samples["precision@"+str(i)]+=precision.item()
                    samples["recall@"+str(i)]+=recall.item()

    for key in samples:
        print(key,samples[key]/len(idxs))

    