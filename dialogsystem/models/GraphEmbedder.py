from torch.nn import Module
from torch.nn.modules.container import ModuleList
from torch.nn import Linear
from torch import cat, tanh, mean
from networkx import compose_all, ego_graph, get_edge_attributes, get_node_attributes
from torch_geometric.utils import from_networkx
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
        if type(text) is str:
            ttext = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
            output = self.encoder(ttext)
            return mean(output[0])
        else:
            texts = [text[i:i+self.max_batchsize] for i in range(0,len(text),self.max_batchsize)]
            outputs = []
            for text in tqdm(texts):
                ttext = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
                outputs.append(self.encoder(ttext)[0])
            return mean(cat(outputs))
        

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
        keys = node_attributes.keys()
        values = [node_attributes[key] for key in keys]
        values = self.lm_embedder(values)
        node_attributes = {key: values[i] for i,key in enumerate(keys)}
        # node_attributes = {key: self.lm_embedder(value) for key,value in node_attributes.items()}

        # for node in node_attributes:
        #     new_node_attributes[node]=self.lm_embedder(node_attributes[node])
        edge_attributes = get_edge_attributes(nxgraph,'label')
        keys = edge_attributes.keys()
        values = [edge_attributes[key] for key in keys]
        values = self.lm_embedder(values)
        edge_attributes = {key: values[i] for i,key in enumerate(keys)}
        # edge_attributes = {key: self.lm_embedder(value) for key, value in edge_attributes.items()}
        # for edge in edge_attributes:
        #     new_edge_attributes[edge]=self.lm_embedder(edge_attributes[edge])
        return node_attributes,edge_attributes        

    def forward(self,nxgraph):
        '''
        arguments:
            nxgraph: a networkx (knowledge) graph
        returns:
            data, edge_index (torch_geometric format)
        Changes the networkx graph into a line graph.

        use the edge_index to reverse the line graph transform (for checking the text on the original triple).
        '''
        node_attributes, edge_attributes = self.embed(nxgraph)
        for edge in edge_attributes: # combine embeddings for nodes and edges
            edge_attributes[edge]=cat[edge_attributes[edge],node_attributes[edge[0]],node_attributes[edge[1]]]
        data = from_networkx(nxgraph,None,edge_attributes)
        edge_index = data.edge_index
        data = LineGraph()(data)
        return data, edge_index

    def add_query(self, graph_data, query):
        query_embedding = self.lm_embedder(query)
        graph_data.x = cat([graph_data.x,query_embedding.expand((graph_data.x.size(0),-1))], dim=1) ## concats on query embedding
        return graph_data