from torch.nn import Module
from torch.nn.modules.container import ModuleList
from torch.nn import Linear
from torch import cat, tanh, mean, no_grad, tensor
from networkx import compose_all, ego_graph, get_edge_attributes, get_node_attributes, set_edge_attributes, set_node_attributes, line_graph, is_directed
from torch_geometric.utils import from_networkx, remove_self_loops
from torch_geometric.transforms import LineGraph
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

class LMEmbedder(Module):
    def __init__(self, lm, tokenizer, max_batchsize=128):
        '''
        Used to embed text using a language model. This is based off of 'Sentence-T5: Scalable Sentence Encoders
        from Pre-trained Text-to-Text Models', Ni et al. 2021
        Uses the encoder of the input encoder-decoder language model to create an encoding by taking th emean of the encoder outputs across all input tokens
        Arguments:
            lm: a hugging face transformers Encoder-Decoder model
            tokenizer: the tokenizer
        Note that this is intended to be frozen!
        '''
        super(LMEmbedder,self).__init__()
        self.encoder = lm
        self.tokenizer = tokenizer
        self.max_batchsize=max_batchsize

    def forward(self, text):
        with no_grad():
            if type(text) is str:
                ttext = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
                output = self.encoder(ttext)
                return mean(output[0],dim=1)
            else:
                ttext = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
                ttexts = [ttext[i:i+self.max_batchsize] for i in range(0,len(ttext),self.max_batchsize)]
                outputs = []
                for ttext in tqdm(ttexts):
                    outputs.append(self.encoder(ttext)[0])
                out = mean(cat(outputs),dim=1)
                return out
        

class JointEmbedder(Module):
    '''
    Used to embed a (query, graph_text) pair into an embedding_size dimensional vector\\
    The graph_text is either the text on the node or on the edges\\
    Note that query and graph_text are embedded into initial_embedding_size (the output of an LM)    
    '''
    def __init__(self, initial_embedding_size,embedding_size,hidden_dim):
        super(JointEmbedder,self).__init__()
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

def _switch_dict(attribute_dict):
    """
    args:
        attribute_dict: {(u,v):atts}
    returns:
        {(v,u):atts}
    """
    out_dict = {}
    for k,v in attribute_dict.items():
        out_dict[(k[1],k[0])]=v
    return out_dict

class GraphTransformer(Module):
    def __init__(self,lm_string="google/byt5-small") -> None:
        """
        lm_string: name of encoder decoder language model used for embedding
        """
        super(GraphTransformer,self).__init__()
        lmodel = T5ForConditionalGeneration.from_pretrained(lm_string).encoder
        tokenizer = AutoTokenizer.from_pretrained(lm_string)
        self.lm_embedder = LMEmbedder(lmodel,tokenizer)

    def embed(self, nxgraph):
        '''
        Returns the node and edge attributes of the result of embedding the graph 
        Arguments:
            graph: networkx graph instance,
            query: string, query used
        '''
        # node_attributes = get_node_attributes(nxgraph,'label')
        node_attributes = {node:node for node in nxgraph.nodes} ## due to error not saving node attributes as labels
        # keys = node_attributes.keys()
        # values = [node_attributes[key] for key in keys]
        # values = self.lm_embedder(values)
        # node_attributes = {key: values[i] for i,key in enumerate(keys)}
        # node_attributes = {key: self.lm_embedder(value) for key,value in node_attributes.items()}

        # for node in node_attributes:
        #     new_node_attributes[node]=self.lm_embedder(node_attributes[node])
        edge_attributes = get_edge_attributes(nxgraph,'label')
        keys = edge_attributes.keys()
        values = [f" {node_attributes[key[0]]}, {edge_attributes[key]}, {node_attributes[key[1]]}" for key in keys]
        values = self.lm_embedder(values)
        edge_attributes = {key: values[i] for i,key in enumerate(keys)}
        # edge_attributes = {key: self.lm_embedder(value) for key, value in edge_attributes.items()}
        # for edge in edge_attributes:
        #     new_edge_attributes[edge]=self.lm_embedder(edge_attributes[edge])
        return edge_attributes        

    def forward(self,nxgraph):
        '''
        arguments:
            nxgraph: a networkx (knowledge) graph
        returns:
            data, edge_index (torch_geometric format)
        Changes the networkx graph into a line graph.

        use the edge_index to reverse the line graph transform (for checking the text on the original triple).
        '''
        edge_attributes = self.embed(nxgraph)
        for edge in edge_attributes:
            edge_attributes[edge] = edge_attributes[edge].tolist()
        # for edge in edge_attributes: # combine embeddings for nodes and edges
        #     edge_attributes[edge]=cat([edge_attributes[edge],node_attributes[edge[0]],node_attributes[edge[1]]]).tolist()
        #    # set_edge_attributes(nxgraph,edge_attributes,'embedding')
        #    # lg = line_graph(nxgraph)
        set_edge_attributes(nxgraph,edge_attributes,'embedding')
        if not is_directed(nxgraph):
            set_edge_attributes(nxgraph,_switch_dict(edge_attributes),'embedding')
        rel_attributes = get_edge_attributes(nxgraph,'relevance_label')
        set_edge_attributes(nxgraph,rel_attributes,'relevance_label')
        if not is_directed(nxgraph):
            set_edge_attributes(nxgraph,_switch_dict(rel_attributes),'relevance_label')
        data = from_networkx(nxgraph,group_edge_attrs=['embedding','relevance_label'])
        if not is_directed(nxgraph):
            data.edge_index,data.edge_attr=remove_self_loops(data.edge_index,data.edge_attr)
        edge_index = data.edge_index
        data = LineGraph()(data)
        return data, edge_index

    def add_query(self, graph_data, query):
        query_embedding = self.lm_embedder(query)
        graph_data.x = cat([graph_data.x,query_embedding.expand((graph_data.x.size(0),-1))], dim=1) ## concats on query embedding
        return graph_data

if __name__ == '__main__':
    from networkx import DiGraph, Graph
    nxgraph = Graph()
    nxgraph.add_nodes_from(["hi","hello","welcome","greetings","don't","ghosted"])
    nxgraph.add_edge("hi","hello",label="is_same",relevance_label=4)
    nxgraph.add_edge("welcome","hi",label="is_same",relevance_label=2)
    nxgraph.add_edge("welcome","greetings",label="don't",relevance_label=4)
    nxgraph.add_edge("hello","hi",label="is_same",relevance_label=3)
    nxgraph.add_edge("hi","hi",label="uh-oh",relevance_label=5)
    nxgraph.add_edge("don't","don't",label="welp",relevance_label=2)
    nxgraph.add_edge("ghosted","ghosted",label="hmm",relevance_label=4)
    nxgraph.add_edge("greetings","greetings",label="I need osmebody", relevance_label=2)
    gt = GraphTransformer(lm_string="t5-small")
    graph,_ = gt(nxgraph)
    graph = gt.add_query(graph, "help")
    print(graph)
    print(graph.x)