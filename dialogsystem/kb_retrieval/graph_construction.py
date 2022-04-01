from multiprocessing import Pool
from networkx import subgraph, single_source_shortest_path_length, compose
from networkx import Graph

def extract_nodes(tup):
    nxg,ec,k = tup
    lengths = single_source_shortest_path_length(nxg, ec, cutoff=k)
    return list(lengths.keys())

class GraphConstructor():
    def __init__(self) -> None:
        self.graph = Graph()

    def input_nx_graph_with_trimming(self,nxgraph,entity_candidates,k):
        pool = Pool()
        nodes = pool.map(extract_nodes,[(nxgraph,ec,k) for ec in entity_candidates])
        pool.close()
        pool.join()
        nodes = [node for nodelist in nodes for node in nodelist]
        self.graph = compose(self.graph,subgraph(nxgraph,nodes))

    def extract_relevant_nodes(self,nxgraph,background_entities,k=1):
        '''
        this method is for getting the label data for any potentially relevant entities
        '''
        pool = Pool()
        nodes = pool.map(extract_nodes,[(nxgraph,ec,k) for ec in background_entities])
        pool.close()
        pool.join()
        nodes = [node for nodelist in nodes for node in nodelist]
        return nodes

    def add_from_triple(self,triple):
        '''
        Builds a networkx graph using a set of triples
        '''
        self.graph.add_node(triple[0]["id"],label=triple[0]["label"])
        self.graph.add_node(triple[2]["id"],label=triple[2]["label"])
        self.graph.add_edge(triple[0]["id"],triple[2]["id"],label=triple[1]["label"])

    def build_graph(self):
        return self.graph