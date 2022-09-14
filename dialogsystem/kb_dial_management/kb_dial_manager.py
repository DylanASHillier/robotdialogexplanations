import string
from typing import Optional
from matplotlib import pyplot as plt
from networkx import MultiDiGraph, get_edge_attributes, read_gpickle, write_gpickle, draw
import json
import sys
from os.path import dirname
from torch import topk
from torch_geometric.data import Data

sys.path.append(dirname("./dialogsystem"))

from dialogsystem.kb_retrieval.candidate_trimming import CandidateGenerator
from dialogsystem.kb_retrieval.graph_construction import GraphConstructor
from dialogsystem.models.GraphEmbedder import GraphTransformer
from dialogsystem.models.triples2text import Triples2TextSystem
from dialogsystem.models.kgqueryextract import LightningKGQueryMPNN
from dialogsystem.models.convqa import ConvQASystem


class DialogueKBManager():
    '''
    Abstract Base Class to be implemented for specific use cases
    includes management and explanation of dialogue

    To use:
        implement initialise_kbs
    '''
    def __init__(self, knowledge_base_args, mpnn, convqa, triples2text, top_k=10, top_p=0.5) -> None:
        '''
        knowledge_bases: dict of knowledge base arguments used by `initialise_kbs` to create the knowledge bases
        mpnn: module that runs the message passing neural network
        convqa: module that runs the convolutional question answering model
        triples2text: module that runs the triples to text model
        top_k: number of triples to extract from the knowledge base
        top_p: sum of probabilities to extract from the knowledge base (used after top_k)
        '''
        self.kbs = []
        self.convqa = convqa
        self.mpnn = mpnn
        self.triples2text = triples2text
        self.dialogue_graph = MultiDiGraph()
        self.dialogue_context = ""
        self.candidategenerator = CandidateGenerator(10,0.3)
        self.graph_constructor = GraphConstructor()
        self.entity_queue = []
        self.entity_queue_max = 100
        self.graph_transformer = GraphTransformer(lm_string="t5-small")
        self._reset()
        self.kbs = self.initialise_kbs(**knowledge_base_args)
        self.kbs = [self._pre_process_graph(kb) for kb in self.kbs]
        self.top_k=top_k
        self.top_p=top_p
        self.logs = {
            'questions': [],
            'base_graphs': self.kbs+[self.dialogue_graph],
            'extracted_graphs': [],
            'extracted_triples': [],
            'extracted_text': [],
            'extracted_answers': [],
            'extracted_context': [],
        }
        self.turn_tracker = 0

    def _reset(self):
        self.dialogue_graph = MultiDiGraph()
        self.dialogue_context = ""
        self.entity_queue = []
        self.turn_tracker = 0

    def initialise_kbs(self, **knowledge_base_args) -> list[MultiDiGraph]:
        '''
        sets up self.kbs
        '''
        return []

    def save_logs(self, folder):
        # save base graphs to json
        for i,graph in enumerate(self.logs["base_graphs"]):
            write_gpickle(graph,f"{folder}/basegraph{i}.json")
        # save base graphs as pictures
        for i,graph in enumerate(self.logs["base_graphs"]):
            draw(graph,with_labels=True)
            plt.savefig(f"{folder}/basegraph{i}.png")
            plt.close()
        # save extracted graphs to json
        for i,graph in enumerate(self.logs["extracted_graphs"]):
            write_gpickle(graph,f"{folder}/extractedgraph{i}.json")
        # save extracted graphs as pictures
        for i,graph in enumerate(self.logs["extracted_graphs"]):
            draw(graph,with_labels=True)
            plt.savefig(f"{folder}/extractedgraph{i}.png")
            plt.close()
        # save remaining data to json, skipping base graphs and extracted graphs
        data = {
            'questions': self.logs['questions'],
            'extracted_triples': self.logs['extracted_triples'],
            'extracted_text': self.logs['extracted_text'],
            'extracted_answers': self.logs['extracted_answers'],
            'extracted_context': self.logs['extracted_context'],
        }
        with open(f"{folder}/data.json", 'w') as f:
            json.dump(data, f)




    def _pre_process_graph(self, graph):
        return self.graph_transformer.update(MultiDiGraph(),graph)

    def _update_dialogue_graph(self, triples, question, answer, extracted_text):
        update = MultiDiGraph()
        turn = f"turn: {self.turn_tracker}"
        # for triple in triples:
        #     update.add_edge(extracted_text,','.join(triple),label='extracted from')
        #     update.add_edge(','.join(triple),turn, label='extracted in')
        # update.add_edge(question,turn,label="asked in")
        # update.add_edge(answer,turn,label="answered in")
        # update.add_edge(extracted_text,turn,label="extracted text in")
 
        if self.turn_tracker>0:
            update.add_edge(turn,f"turn: {self.turn_tracker-1}",label="previous turn")
            update.add_edge(f"turn: {self.turn_tracker-1}",turn,label="next turn")
        else:
            update.add_edge(turn,"first turn",label="is")
            update.add_edge("first turn",turn,label="is")
        self.dialogue_graph = self.graph_transformer.update(self.dialogue_graph, update)

    def question_and_response(self,question: str):
        self.logs['questions'].append(question)
        graph = self._extract_relevant_graph_from_query(question)
        self.logs['extracted_graphs'].append(graph)
        data = self._transform_graph(graph, question)
        triples = self._run_mpnn(data)
        self.logs['extracted_triples'].append(triples)
        triples = [triple.split(",") for triple in triples]
        text = self._run_triples2text(triples)
        self.logs['extracted_text'].append(text)
        answer = self._run_convqa(question, text)
        self.logs['extracted_answers'].append(answer)
        # self._update_dialogue_graph(triples, question, answer, text)
        self._update_dialogue_context(question, answer)
        self.turn_tracker += 1
        return answer

    def _update_dialogue_context(self,question, answer):
        self.dialogue_context += f"{question} -> {answer}\n"

    def _get_entities_from_graph(self, graph):
        all_entities= graph.nodes.items()
        all_entities = {entity[0]:entity[0] for entity in all_entities}
        return all_entities
        
    def _extract_relevant_graph_from_query(self, question: str):
        all_entities = self._get_entities_from_graph(self.dialogue_graph)
        all_entities = self.candidategenerator.transformer_trim(question,all_entities)
        self.graph_constructor.input_nx_graph_with_trimming(self.dialogue_graph, all_entities, 3)
        for kb in self.kbs:
            entities = self._get_entities_from_graph(kb)
            entities = self.candidategenerator.transformer_trim(question,entities)
            self.graph_constructor.input_nx_graph_with_trimming(kb, entities, 3)
            all_entities += entities
        return self.graph_constructor.build_graph()
    
    def _transform_graph(self, graph: Optional[MultiDiGraph], question: string)-> Optional[Data]:
        if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
            return None
        data = self.graph_transformer.transform(graph, False)
        data = self.graph_transformer.add_query(data, question, relevance_label=False)
        return data

    def _run_mpnn(self, data: Optional[Data]):
        if data is None:
            return []
        output = self.mpnn(data.x, data.edge_index)
        output, indices = topk(output, min(self.top_k, output.size(0)),sorted=True)
        print(indices)
        print("Output:", output)
        # output until sum reaches top_p
        running_sum = 0
        for i,e in enumerate(output):
            running_sum += e
            if running_sum >= self.top_p:
                indices = indices[:i+1]
                break


        triples = [data.edge_label[idx] for idx in indices]
        return triples

    def _coalesce_triples(self,triples):
        '''
        given a list of triples, concatenate any that have the same entity
        '''
        def remove_self_loop(triple):
            if triple[0] == triple[2]:
                return triple[:2]
            else:
                return triple
        def search(triple, coalesced_triples) -> bool:
            for coalesced_triple in coalesced_triples:
                for other_triples in coalesced_triple:
                    if triple[0] in other_triples or triple[2] in other_triples:
                        coalesced_triple.append(remove_self_loop(triple))
                        return True
            return False

        coalesced_triples = []
        for triple in triples:
            if search(triple, coalesced_triples):
                continue
            coalesced_triples.append([remove_self_loop(triple)])
        return coalesced_triples
            

    def _run_triples2text(self,triples):
        coalesced_triples = self._coalesce_triples(triples)
        coalesced_triples = [[",".join(triple) for triple in coalesced_triple] for coalesced_triple in coalesced_triples]
        triples = [";".join(coalesced_triple) for coalesced_triple in coalesced_triples]
        if len(triples) == 0:
            return ""
        return '\n'.join(self.triples2text(triples))

    def _run_convqa(self,question, background_text):
        prompt = f"background: {background_text}\n context: {self.dialogue_context}\n question: {question}"
        # print(prompt, self.dialogue_context, question)
        answer = self.convqa(prompt)
        return answer


if __name__ == "__main__":
    from random import sample, choice
    convqa = ConvQASystem("./dialogsystem/trained_models/convqa")
    triples2text = Triples2TextSystem("./dialogsystem/trained_models/t2t/t2ttrained")
    mpnn = LightningKGQueryMPNN.load_from_checkpoint("dialogsystem/trained_models/gqanew.ckpt")
    mpnn.k = 3
    kb_manager = DialogueKBManager({},mpnn,convqa,triples2text)
    print(kb_manager._coalesce_triples([]))
    print(kb_manager._coalesce_triples([("a","b","c"),("a","b","d"),("e","f","g"),("h","f","g")]))
    empty_graph = MultiDiGraph()
    tiny_graph = MultiDiGraph()
    tiny_graph.add_node("banana")  
    small_graph = MultiDiGraph()
    small_graph.add_edge("banana","fruit",label="is a")
    small_graph.add_node("apple")

    sample_nodes = ["banana","fruit","apple","orange","pear","grape","chair","table","fridge","catalyst","mandarin","mercury"]
    sample_edge_labels = ["is a","has","wants","fights","is in","is on"]

    medium_graph = MultiDiGraph()
    for i in range(1000):
        medium_graph.add_edge(choice(sample_nodes),choice(sample_nodes),label=choice(sample_edge_labels))

    triples = [
        ("arachnid","next to","banana"),
        ("banana","goes in","fridge"),
        ("fridge","powered by","electricity"),
        ("banana","is a","fruit"),
        ("spider","is a","arachnid"),
        ("spider","eats","banana"),
        ("spider","eats","arachnid"),
        ("eagles","eat","electricity"),
        ("spider","powered by","arachnid"),
        ("spider","goes in","arachnid")
        ]
    kb_manager.kbs = [empty_graph]
    print("testing empty graph")
    print(kb_manager.question_and_response("where is the banana?"))
    print(kb_manager.question_and_response("what about the spider?"))

    kb_manager.kbs = [kb_manager._pre_process_graph(small_graph)]
    print("testing small graph")
    print(kb_manager.question_and_response("where is the banana?"))
    print(kb_manager.question_and_response("what about the spider?"))

    kb_manager.kbs = [kb_manager._pre_process_graph(tiny_graph)]
    print("testing tiny graph")
    print(kb_manager.question_and_response("where is the banana?"))
    print(kb_manager.question_and_response("what about the spider?"))


    graph = MultiDiGraph(read_gpickle("datasets/KGs/conceptnet.json"))
    random_nodes = sample(list(graph.nodes),10000)
    large_graph = graph.subgraph(random_nodes)

    kb_manager.kbs = [kb_manager._pre_process_graph(large_graph)]
    print("testing large graph")
    print(kb_manager.question_and_response("where is the banana?"))
    print(kb_manager.question_and_response("what about the spider?"))

    kb_manager.kbs = [kb_manager._pre_process_graph(medium_graph)]
    print("testing medium graph")
    print(kb_manager.question_and_response("where is the banana?"))
    print(kb_manager.question_and_response("what about the spider?"))

    # tests multiple graphs
    kb_manager.kbs = [kb_manager._pre_process_graph(large_graph),kb_manager._pre_process_graph(small_graph)]
    print("testing multiple graphs")
    print(kb_manager.question_and_response("where is the banana?"))
    print(kb_manager.question_and_response("what about the spider?"))

    # test mpnn output
    print(kb_manager.logs["extracted_triples"])

    # test triples2text capabilities
    print(kb_manager.logs["extracted_text"])

