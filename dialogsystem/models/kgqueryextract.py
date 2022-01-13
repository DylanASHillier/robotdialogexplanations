from torch.nn import Module, ModuleList
from torch_geometric.nn import GATv2Conv
from torch.nn import Conv1d
from torch import sigmoid, topk

class KGQueryMPNN(Module):
    def __init__(self, num_layers, embedding_size, k):
        '''
        Operates over a graph to obtain the indices of edges that should be used in the knowledge graph
        '''
        self.final_layer = Conv1d(embedding_size,1,1)
        self.hidden_layers = ModuleList([GATv2Conv(embedding_size,embedding_size,edge_dim=embedding_size) for i in range(num_layers)])
        self.k = k

    def forward(self, x, edge_index, edge_attributes):
        for layer in self.hidden_layers:
            x = layer(x,edge_index, edge_attributes)
        x = self.final_layer(x)
        return topk(x,10)
