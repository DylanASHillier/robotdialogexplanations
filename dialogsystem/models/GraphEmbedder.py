from torch.nn import Module
from torch.nn.modules.container import ModuleList
from torch.nn import Linear
from torch import cat, tanh, mean
from networkx import compose_all, ego_graph, get_edge_attributes, get_node_attributes
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import LineGraph

class LMEmbedder(Module):
    '''
    Used to embed text using a language model. This is based off of 'Sentence-T5: Scalable Sentence Encoders
    from Pre-trained Text-to-Text Models', Ni et al. 2021
    Uses the encoder of the input encoder-decoder language model to create an encoding by taking th emean of the encoder outputs across all input tokens
    Arguments:
        lm: a hugging face transformers Encoder-Decoder model
        tokenizer: the tokenizer
    Note that this is intended to be frozen!
    '''
    def __init__(self, lm, tokenizer):
        super().__init__()
        self.encoder = lm
        self.tokenizer = tokenizer

    def forward(self, text):
        ttext = self.tokenizer(text)
        output = self.encoder(ttext)
        return mean(output[0])
        

class JointEmbedder(Module):
    '''
    Used to embed a (query, graph_text) pair into an embedding_size dimensional vector\\
    The graph_text is either the text on the node or on the edges\\
    Note that query and graph_text are embedded into initial_embedding_size (the output of an LM)    
    '''
    def __init__(self, initial_embedding_size,embedding_size,hidden_dim):
        super().__init__()
        self.layers = ModuleList()
        self.layers.append(Linear(initial_embedding_size*2,hidden_dim))
        for i in range(2):
            self.layers.append(Linear(hidden_dim,hidden_dim))
        self.layers.append(Linear(hidden_dim,embedding_size))

    def forward(self, graph_texts, query):
        x = cat([graph_texts,query])
        for layer in self.layers:
            x = tanh(layer(x))
        return x

class GraphTransformer(Module):
    def __init__(self,) -> None:
        self.graph_embedder = GraphDataEmbedder()

    def forward(self,nxgraph):
        '''
        arguments:
            nxgraph: a networkx (knowledge) graph
        returns:
            data, edge_index (torch_geometric format)
        Changes the networkx graph into a line graph.

        use the edge_index to reverse the line graph transform (for checking the text on the original triple).
        '''
        node_attributes, edge_attributes = self.graph_embedder(nxgraph)
        for edge in edge_attributes: # combine embeddings for nodes and edges
            edge_attributes[edge]=cat[edge_attributes[edge],node_attributes[edge[0]],node_attributes[edge[1]]]
        data = from_networkx(nxgraph,None,edge_attributes)
        edge_index = data.edge_index
        data = LineGraph()(data)
        return data, edge_index
        
class GraphDataEmbedder(Module):
    def __init__(self, lm_embedding_size, lm_embedder, embedding_size):
        super().__init__()
        self.lm=lm_embedder
        self.nodeEmbedder = JointEmbedder(lm_embedding_size, embedding_size)
        self.edgeEmbedder = JointEmbedder(lm_embedding_size, embedding_size)

    def forward(self, graph, query):
        '''
        Arguments:
            graph: networkx graph instance,
            query: string, query used
        '''
        node_attributes = get_node_attributes(graph)
        for node in node_attributes:
            node_attributes[node]=self.node_embedder(node_attributes[node],query)
        edge_attributes = get_edge_attributes(graph)
        for edge in edge_attributes:
            edge_attributes[edge]=self.edge_embedder(edge_attributes[edge],query)
        return node_attributes,edge_attributes