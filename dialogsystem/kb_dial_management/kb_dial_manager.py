from networkx import DiGraph, get_edge_attributes, read_gpickle

import sys
from os.path import dirname

sys.path.append(dirname("./dialogsystem"))

from dialogsystem.kb_retrieval.candidate_trimming import CandidateGenerator
from dialogsystem.kb_retrieval.graph_construction import GraphConstructor
from dialogsystem.models.GraphEmbedder import GraphTransformer
from dialogsystem.models.triples2text import Triples2TextSystem
from dialogsystem.models.kgqueryextract import LightningKGQueryMPNN


class DialogueKBManager():
    '''
    Abstract Base Class to be implemented for specific use cases
    includes management and explanation of dialogue

    To use:
        implement initialise_kbs
    '''
    def __init__(self,knowledge_base_args,mpnn,convqa,triples2text) -> None:
        '''
        knowledge_bases: dict of knowledge base arguments used by `initialise_kbs` to create the knowledge bases
        mpnn: module that runs the message passing neural network
        convqa: module that runs the convolutional question answering model
        triples2text: module that runs the triples to text model
        '''
        self.kbs = []
        self.convqa = convqa
        self.mpnn = mpnn
        self.triples2text = triples2text
        self.dialogue_graph = DiGraph()
        self.dialogue_context = ""
        self.candidategenerator = CandidateGenerator(3,0.3)
        self.graph_constructor = GraphConstructor()
        self.entity_queue = []
        self.entity_queue_max = 100
        self.graph_transformer = GraphTransformer(lm_string="t5-small")
        self._reset()
        self.kbs = self.initialise_kbs(**knowledge_base_args)
        self.kbs = [self._pre_process_graph(kb) for kb in self.kbs]

    def _reset(self):
        self.dialogue_graph = DiGraph()
        self.dialogue_context = ""
        self.entity_queue = []

    def initialise_kbs(self, **knowledge_base_args) -> list[DiGraph]:
        '''
        sets up self.kbs
        '''
        return []

    def _pre_process_graph(self, graph):
        return self.graph_transformer.update(DiGraph(),graph)

    def _update_dialogue_graph(self, triples):
        update = DiGraph()
        for triple in triples:
            update.add_edge(triple[0], triple[2], label=triple[1])
        self.dialogue_graph = self.graph_transformer.update(self.dialogue_graph, update)

    def question_and_response(self,question: str):
        graph = self._extract_relevant_graph_from_query(question)
        data = self._transform_graph(graph, question)
        triples = self._run_mpnn(data)
        self._update_dialogue_graph(triples)
        text = self._run_triples2text(triples)
        answer = self._run_convqa(question, text)
        self._update_dialogue_context(question, answer)
        return answer

    def _update_dialogue_context(self,question, answer):
        self.dialogue_context += f"{question} -> {answer}\n"

    def _get_entities_from_graph(self, graph):
        all_entities= graph.nodes.items()
        all_entities = {entity[0]:entity[0] for entity in all_entities}
        return all_entities
        
    def _extract_relevant_graph_from_query(self, question: str):
        all_entities = self._get_entities_from_graph(self.dialogue_graph)
        all_entities = self.candidategenerator.trim(question,all_entities)
        self.graph_constructor.input_nx_graph_with_trimming(self.dialogue_graph, all_entities, 3)
        for kb in self.kbs:
            entities = self._get_entities_from_graph(kb)
            entities = self.candidategenerator.trim(question,entities)
            self.graph_constructor.input_nx_graph_with_trimming(kb, entities, 1)
            all_entities += entities
        self._update_dialogue_graph([(entity,"candidate match for", question) for entity in entities])
        return self.graph_constructor.build_graph()
    
    def _transform_graph(self,graph, question):
        # self.processed_graph = self.graph_transformer.update(self.processed_graph, graph, False)
        print(graph)
        data = self.graph_transformer.transform(graph, False)
        data = self.graph_transformer.add_query(data, question)
        return data

    def _run_mpnn(self,data):
        topk = self.mpnn(data.x, data.edge_index).tolist()
        triples = [data.edge_label[idx] for idx in topk]
        return triples

    def _run_triples2text(self,triples):
        print(triples)
        return self.triples2text(triples)

    def _run_convqa(self,question, background_text):
        prompt = f"background: {background_text}\n context: {self.dialogue_context}\n question: {question}"
        print(prompt, self.dialogue_context, question)
        answer = self.convqa(prompt)
        return answer


if __name__ == "__main__":
    convqa = lambda x: "i'm not sure"
    triples2text = Triples2TextSystem.load_from_checkpoint("dialogsystem/trained_models/t2t.ckpt")
    mpnn = LightningKGQueryMPNN.load_from_checkpoint("dialogsystem/trained_models/meddim.ckpt")
    mpnn.k = 3
    kb_manager = DialogueKBManager([],mpnn,convqa,triples2text)
    from random import sample
    graph = read_gpickle("datasets/KGs/conceptnet.json")
    random_nodes = sample(list(graph.nodes),10000)
    kb_manager.kbs=[kb_manager._pre_process_graph(graph.subgraph(random_nodes))]
    # kb_manager.dialogue_graph.add_edge("arachnid","banana",label="next to")
    # kb_manager.dialogue_graph.add_edge("banana","fridge",label="goes in")
    # kb_manager.dialogue_graph.add_edge("fridge","electricity",label="powered by")
    # kb_manager.dialogue_graph.add_edge("banana","fruit",label="is a")
    # kb_manager.dialogue_graph.add_edge("spider","arachnid",label="is a")
    # kb_manager.dialogue_graph.add_edge("spider","banana",label="eats")
    # kb_manager.dialogue_graph.add_edge("eagles","electricity",label="eat")
    triples = [
        ("arachnid","next to","banana"),
        ("banana","goes in","fridge"),
        ("fridge","powered by","electricity"),
        ("banana","is a","fruit"),
        ("spider","is a","arachnid"),
        ("spider","eats","banana"),
        ("eagles","eat","electricity")
        ]
    kb_manager._update_dialogue_graph(triples)
    print(kb_manager.question_and_response("where is the banana?"))
    print(kb_manager.question_and_response("what about the spider?"))


