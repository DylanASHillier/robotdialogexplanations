from torch.nn import Module
from torch.nn.modules.container import ModuleList
from torch.nn import Linear, MSELoss
from torch.optim import Adam
from torch import cat, tanh, mean, no_grad, tensor
from networkx import compose, ego_graph, get_edge_attributes, get_node_attributes, set_edge_attributes, set_node_attributes, line_graph, is_directed
from torch_geometric.utils import from_networkx, remove_self_loops
from torch_geometric.transforms import LineGraph
from transformers import T5ForConditionalGeneration, AutoTokenizer
from pytorch_lightning import LightningModule
from tqdm import tqdm
from os import path

class LitAutoEncoder(LightningModule):
    def __init__(self, init_dim=512, out_dim=25, lr=1e-3):
        super(LitAutoEncoder,self).__init__()
        self.encoder = Linear(init_dim, out_dim)
        self.decoder = Linear(out_dim, init_dim)
        self.loss = MSELoss()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        x = self.encoder(x)
        return x

    def training_step(self, x):
        y = x
        x = self.encoder(y)
        pred = self.decoder(x)
        loss = self.loss(pred, y)
        self.log("train_loss",loss.item())
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class LMEmbedder(Module):
    def __init__(self, lm, tokenizer, max_batchsize=128, auto_encoder=None):
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
        if auto_encoder is not None:
            self.auto_encoder = auto_encoder

    def forward(self, text):
        with no_grad():
            if type(text) is str:
                ttext = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
                output = self.encoder(ttext)
                output = mean(output[0],dim=1)
            else:
                ttext = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
                ttexts = [ttext[i:i+self.max_batchsize] for i in range(0,len(ttext),self.max_batchsize)]
                outputs = []
                for ttext in tqdm(ttexts):
                    outputs.append(self.encoder(ttext)[0])
                output = mean(cat(outputs),dim=1)
            if self.auto_encoder is not None:
                output = self.auto_encoder(output)
        return output
        

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
        Class for handling embedding of networkx graph with a language model, transformation to torch_geometric, and query embedding
        Arguments:
            lm_string: name of encoder decoder language model used for embedding the query and graph
        """
        super(GraphTransformer,self).__init__()
        lmodel = T5ForConditionalGeneration.from_pretrained(lm_string).encoder
        tokenizer = AutoTokenizer.from_pretrained(lm_string)
        if path.exists("dialogsystem/trained_models/autoencoder.ckpt"):
            auto_encoder = LitAutoEncoder.load_from_checkpoint("dialogsystem/trained_models/autoencoder.ckpt")
        else:
            auto_encoder =None
        self.lm_embedder = LMEmbedder(lmodel,tokenizer, auto_encoder= auto_encoder)

    def embed(self, nxgraph):
        '''
        Returns the node and edge attributes of the result of embedding the graph 
        Arguments:
            graph: networkx graph instance,
            query: string, query used
        '''
        node_attributes = {node:node for node in nxgraph.nodes}
        edge_attributes = get_edge_attributes(nxgraph,'label')
        keys = edge_attributes.keys()
        values = [f" {node_attributes[key[0]]}, {edge_attributes[key]}, {node_attributes[key[1]]}" for key in keys]
        edge_labels = {key: values[i] for i,key in enumerate(keys)}
        values = self.lm_embedder(values)
        edge_attributes = {key: values[i].tolist() for i,key in enumerate(keys)}
        return edge_attributes, edge_labels

    def _update_attributes(self, nxgraph, edge_attributes, edge_labels, relevance_label=False):
        '''
        updates the edge attributes of the graph with the new embedded edge attributes
        Arguments:
            nxgraph: networkx graph instance,
            edge_attributes: {(u,v):atts},
            relevance_labels: bool, if true, the edge attributes are updated with the relevance labels
        ''' 
        set_edge_attributes(nxgraph,edge_attributes,'embedding')
        set_edge_attributes(nxgraph,edge_labels,'edge_label')
        if not is_directed(nxgraph):
            set_edge_attributes(nxgraph,_switch_dict(edge_attributes),'embedding')
            set_edge_attributes(nxgraph,_switch_dict(edge_labels),'edge_label')
        if relevance_label:
            rel_attributes = get_edge_attributes(nxgraph,'relevance_label')
            set_edge_attributes(nxgraph,rel_attributes,'relevance_label')
            if not is_directed(nxgraph):
                set_edge_attributes(nxgraph,_switch_dict(rel_attributes),'relevance_label')
        return nxgraph

    def transform(self, nxgraph, relevance_label=False):
        '''
        Transforms the graph into a line graph from networkx
        Arguments:
            nxgraph: networkx graph instance,
            relevance_label: boolean, whether to add relevance labels to the graph
        '''
        if relevance_label:
            data = from_networkx(nxgraph,group_edge_attrs=['embedding', 'relevance_label'])
        else:
            data = from_networkx(nxgraph,group_edge_attrs=['embedding'])
        if not is_directed(nxgraph):
            data.edge_index,data.edge_attr=remove_self_loops(data.edge_index,data.edge_attr)
        data = LineGraph()(data)
        return data

    def update(self, original_graph, new_graph, relevance_label=False):
        '''
        used to update the graph with new triples
        Arguments:
            original_graph: networkx graph instance,
            new_graph: networkx graph instance,
            relevance_label: boolean, whether to utilise relevance labels as attributes
        '''
        new_graph = compose(original_graph,new_graph)
        edge_attributes, edge_labels = self.embed(new_graph)
        new_graph = self._update_attributes(new_graph, edge_attributes, edge_labels, relevance_label)
        return new_graph

    def forward(self,nxgraph, relevance_label=True):
        '''
        arguments:
            nxgraph: a networkx (knowledge) graph,
            relevance_label: boolean, whether to utilise relevance labels
        returns:
            data, edge_index (torch_geometric format)
        Changes the networkx graph into a line graph.

        use the edge_index to reverse the line graph transform (for checking the text on the original triple).
        '''
        edge_attributes, edge_labels = self.embed(nxgraph)
        nxgraph = self._update_attributes(nxgraph, edge_attributes, edge_labels, relevance_label = relevance_label)
        return self.transform(nxgraph, relevance_label = relevance_label)

    def add_query(self, graph_data, query):
        query_embedding = self.lm_embedder(query)
        graph_data.x = cat([graph_data.x,query_embedding.expand((graph_data.x.size(0),-1))], dim=1) ## concats on query embedding
        return graph_data

if __name__ == '__main__':
    from networkx import DiGraph, Graph
    nxgraph = DiGraph()
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
    nxgraph = gt._update_attributes(nxgraph,gt.embed(nxgraph))
    newgraph = DiGraph()
    newgraph.add_nodes_from(["hi","salut"])
    newgraph.add_edge("hi","salut",label="is_same")
    newgraph = gt.update(nxgraph,newgraph)
    graph, edge_index = gt.transform(newgraph,relevance_label=False)
    graph = gt.add_query(graph, "help")
    print(graph)
    print(graph.x)