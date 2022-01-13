from torch.nn import Module

class JointNodeEmbedder(Module):
    '''
    Used to embed a (query, node_text) pair into an embedding_size dimensional vector
    '''
    def __init__(self, initial_embedding_size,embedding_size):
        super().__init__()

class JointEdgeEmbedder(Module):
    '''
    Used to embed a (query, edge_text) pair into an embedding_size dimensional vector
    '''

class GraphDataEmbedder(Module):
    def __init__(self, initial_embedding_size,embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, nodes, edge_attributes, query):
        updated_node_representations = 