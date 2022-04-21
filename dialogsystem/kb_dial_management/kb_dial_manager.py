from networkx import DiGraph, get_edge_attributes

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
    '''
    def __init__(self,knowledge_bases,mpnn,convqa,triples2text) -> None:
        '''
        knowledge_bases: list of knowledge bases to use
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
        self.processed_graph = DiGraph()
        self._reset()
        self.initialise_kbs(knowledge_bases)

    def _reset(self):
        self.dialogue_graph = DiGraph()
        self.dialogue_context = ""
        self.entity_queue = []
        self.processed_graph = DiGraph()

    def initialise_kbs(self, knowledge_base_args):
        '''
        sets up self.kbs
        '''
        pass

    def question_and_response(self,question: str):
        entities = self._entities_from_query(question)
        graph = self._extract_graph(entities)
        data = self._transform_graph(graph, question)
        triples = self._run_mpnn(data)
        for triple in triples:
            self.dialogue_graph.add_edge(f"triple:<{str(triple)}>",question,label="deemed relevent for")
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
        
    def _entities_from_query(self, question: str):
        all_entities = self._get_entities_from_graph(self.dialogue_graph)
        entities = self.candidategenerator.trim(question,all_entities)
        for kb in self.kbs:
            all_entities = self._get_entities_from_graph(kb)
            entities += self.candidategenerator.trim(question,kb)
        for entity in entities:
            self.dialogue_graph.add_edge(entity,question,label="asked about in")
        return entities

    def _extract_graph(self,entities):
        self.graph_constructor.reset()
        for graph in self.kbs:
            self.graph_constructor.input_nx_graph_with_trimming(graph, entities, 3)
        self.graph_constructor.input_nx_graph_with_trimming(self.dialogue_graph, entities, 3)
        graph = self.graph_constructor.build_graph()
        return graph
    
    def _transform_graph(self,graph, question):
        self.processed_graph = self.graph_transformer.update(self.processed_graph, graph, False)
        data = self.graph_transformer.transform(self.processed_graph, False)
        data = self.graph_transformer.add_query(data, question)
        return data

    def _run_mpnn(self,data):
        topk = self.mpnn(data.x, data.edge_index).tolist()
        triples = [data.edge_label[idx] for idx in topk]
        return triples

    def _run_triples2text(self,triples):
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
    kb_manager = DialogueKBManager([],mpnn,convqa,triples2text)
    kb_manager.dialogue_graph.add_edge("triple:<(a,b,c)>","question:<what is a>",label="deemed relevent for")
    kb_manager.dialogue_graph.add_edge("question:<what is a>","triple:<(a,b,c)>",label="asked about in")
    kb_manager.dialogue_graph.add_edge("triple:<(a,b,c)>","question:<what is b>",label="deemed relevent for")
    kb_manager.dialogue_graph.add_edge("question:<what is b>","triple:<(a,b,c)>",label="asked about in")
    kb_manager.dialogue_graph.add_edge("triple:<(a,b,c)>","question:<what is c>",label="deemed relevent for")
    kb_manager.dialogue_graph.add_edge("question:<what is c>","triple:<(a,b,c)>",label="asked about in")
    print(kb_manager.question_and_response("what was relevant when I asked what a was?"))
    print(kb_manager.question_and_response("what about when I just asked?"))


