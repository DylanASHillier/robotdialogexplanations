from networkx import compose_all, ego_graph
from networkx import Graph

class GraphConstructor():
    def __init__(self) -> None:
        self.graph = Graph()

    def input_nx_graph_with_trimming(self,nxgraph,entity_candidates,k):
        self.graph = compose_all([ego_graph(nxgraph,entity,radius=k) for entity in entity_candidates]+[self.graph])

    def add_from_triple(self,triple):
        '''
        Builds a networkx graph using a set of triples
        '''
        self.graph.add_node(triple[0]["id"],label=triple[0]["label"])
        self.graph.add_node(triple[2]["id"],label=triple[2]["label"])
        self.graph.add_edge(triple[0]["id"],triple[2]["id"],label=triple[1]["label"])

    def build_graph(self):
        return self.graph