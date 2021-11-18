from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class KGQueryMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.node_text_embedder = None #TODO encoding for text

    def forward(self, x, edge_index, query):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E, edge_channels]
        # query has shape [?]

        # Step 1: Embed Edge Matrix  with text embeddings as edge_attributes
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Add self loops anyway?

        # Step 2: Embed Feature Matrix with text embeddings.
        x = self.text_embedder(x,query)

        # Step 3: Compute normalization. TODO Should we?
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        edge_labels = self.propagate(edge_index, x=x, norm=norm)

        #TODO Step 6: Add Context/Output judgements to KG (to enable self reflection)
        return output, updated_graph

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j