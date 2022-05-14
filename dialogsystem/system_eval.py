## Script for doing ablation studies on the system as a whole using the gqa dataset.
import sagemaker
import argparse
from kb_retrieval.candidate_trimming import CandidateGenerator
from kb_retrieval.graph_construction import GraphConstructor
from models.GraphEmbedder import GraphTransformer
from models.triples2text import Triples2TextSystem
from models.kgqueryextract import LightningKGQueryMPNN
from models.convqa import ConvQASystem
from torch import topk
from numpy import random
from data.gqaDataset import gqa
from tqdm import tqdm
from networkx import get_edge_attributes
from torchmetrics import SQuAD


graph_transformer = GraphTransformer(lm_string="t5-small")
### define options for each component ###
cg = CandidateGenerator(10,0.8)
graph_trimming_options = {
    "no_trimming": lambda q,ent: ent.values(),
    "trim_candidates": lambda q,ent: cg.trim(q,ent),
}
qtext_mpnn = LightningKGQueryMPNN.load_from_checkpoint("gpuqtextnew.ckpt")
gqa_mpnn =  LightningKGQueryMPNN.load_from_checkpoint("gpugqa2.ckpt")

mpnn_options = {
    "no_mpnn": lambda x, e_index: x[:,-1],
    "gqa_mpnn": lambda x, e_index: gqa_mpnn(x, e_index),
    "qtext_mpnn": lambda x, e_index: qtext_mpnn(x, e_index),
}
# t2t_model = Triples2TextSystem("dialogsystem/trained_models/t2t")
t2t_options = {
    "no_t2t": lambda x: ''.join(x),
    # "t2t": lambda x: t2t_model(x),
}
convqa_model = ConvQASystem("dialogsystem/trained_models/checkpoint-2500")
convqa_options = {
    "no_convqa": lambda background, rest: background,
    "convqa": lambda background, rest: convqa_model(background+rest),
}
background_pref = "background: "
context_pref = "context: "
question_pref = "question: "
target_pref = "answer: "
prefix = "Answer the Question: "

def get_pair(entity):
    '''
    takes a (entity_id,node_attributes) pair. If the node attributes are empty it returns (entity_id,entity_id), otherwise
    it should extract the label
    '''
    if not entity[1]: # checks if the dictionary is empty
        return entity[0], entity[0]
    else:
        return entity[0], entity[1]['label']

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gqa")
    parser.add_argument("--mpnn", type=str, default="no_mpnn")
    parser.add_argument("--trimming", type=str, default="no_trimming")
    parser.add_argument("--t2t", type=str, default="no_t2t")
    parser.add_argument("--convqa", type=str, default="no_convqa")
    return parser.parse_args()

if __name__=='__main__':
    args = arg_parser()
    cg_option = graph_trimming_options[args.trimming]
    mpnn_options = mpnn_options[args.mpnn]
    t2t_options = t2t_options[args.t2t]
    convqa_options = convqa_options[args.convqa]
    gc = GraphConstructor()
    if args.dataset == "gqa":
        ds = gqa("train")
        random.seed(123)
        idxs = random.choice(range(len(ds)), size=1000)
        graph_transformer = GraphTransformer(lm_string="t5-small")
        scorer = SQuAD()

        for i in tqdm(idxs):
            question, answer, graph = ds[i]
            all_entities= graph.nodes.items()
            all_entities = {get_pair(entity)[1]:get_pair(entity)[0] for entity in all_entities}
            all_relations = set(get_edge_attributes(graph,"label").values())
            all_relations = {relation:relation for relation in all_relations}  
            
            # triples, background_entities = query_wikidata(de,question,conv_context,background)
            cg = CandidateGenerator(5,0.8)
            question_entities= cg_option(question,all_entities)
            gc.input_nx_graph_with_trimming(graph, question_entities, 3)
            edge_label_dict = {}
            processed_graph = gc.build_graph()
            if processed_graph.size()>0:
                processed_graph = graph_transformer(processed_graph, relevance_label=False)
                data = graph_transformer.add_query(processed_graph,question, relevance_label=False)
                output = mpnn_options(data.x, data.edge_index)

                choices = topk(output, k=min(5,output.size(0))[1]
                triples = [data.edge_label[choice] for choice in choices]
                triples_to_text = t2t_options(triples)

                predicted_answer = convqa_options(f"{background_pref} {triples_to_text}", f" \n {context_pref} \n {question_pref} {question}")
                squad_prediction = {
                    "prediction_text": predicted_answer,
                    "id": i
                }
            
                squad_target = {
                    "answers": 
                        {
                            "answer_start": [i],
                            "text": [f"{target_pref} {answer}"]
                        }
                    ,
                    "id": i
                }
                scorer.update(squad_prediction, squad_target)
    elif args.dataset == "convqa":
        pass
    print(scorer.compute(), f"for {args.mpnn}, {args.trimming}, {args.t2t}, {args.convqa}")



