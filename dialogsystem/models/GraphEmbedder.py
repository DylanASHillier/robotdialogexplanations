from torch.nn import Module
from torch.nn.modules.container import ModuleList
from torch.nn import Linear, MSELoss
from torch.optim import Adam
from torch import cat, tanh, mean, no_grad, tensor, zeros, LongTensor
from networkx import compose, get_edge_attributes, get_node_attributes, set_edge_attributes, set_node_attributes, line_graph, is_directed, MultiDiGraph
from torch_geometric.utils import from_networkx, remove_self_loops
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from transformers import T5ForConditionalGeneration, AutoTokenizer
from pytorch_lightning import LightningModule
from tqdm import tqdm
from typing import Optional, Union, List, Tuple, Dict
from collections import defaultdict

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
                if len(text) == 0:
                    return zeros((0,))
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
    def __init__(self,lm_string="t5-small", auto_encoder_path="dialogsystem/trained_models/autoencoder.ckpt") -> None:
        """
        Class for handling embedding of networkx graph with a language model, transformation to torch_geometric, and query embedding
        Arguments:
            lm_string: name of encoder decoder language model used for embedding the query and graph

        Methods:
            embed_query: used for adding the query embedding to each node
            embed: used for embedding the graph, returns embedded edge labels
            update: used for updating the embedded graph with new triples
            forward: calls the main transformations, excluding the addtion of the query embedding

        Thus typical usage is just:
        graph_data = gt(graph)
        graph_data = embed_query(graph_data, query)
        """
        super(GraphTransformer,self).__init__()
        lmodel = T5ForConditionalGeneration.from_pretrained(lm_string).encoder
        tokenizer = AutoTokenizer.from_pretrained(lm_string)
        if auto_encoder_path is not None:
            auto_encoder = LitAutoEncoder.load_from_checkpoint(auto_encoder_path)
        else:
            auto_encoder =None
        self.lm_embedder = LMEmbedder(lmodel,tokenizer, auto_encoder= auto_encoder)

    def embed(self, nxgraph: Optional[MultiDiGraph]) -> Optional[Tuple[Dict, Dict]]:
        '''
        Returns the node and edge attributes of the result of embedding the graph 
        Arguments:
            graph: networkx graph instance,
            query: string, query used
        '''
        if nxgraph is None:
            return None
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

    def transform(self, nxgraph: Optional[MultiDiGraph], relevance_label: bool= False) -> Optional[Data]:
        '''
        Transforms the graph into a line graph from networkx
        Arguments:
            nxgraph: networkx graph instance,
            relevance_label: boolean, whether to add relevance labels to the graph
        Returns Pytorch Geom Data or None if the Graph is empty
        '''
        if nxgraph is None or nxgraph.number_of_edges()==0:
            return None
        if relevance_label:
            data = _from_multigraph_networkx(nxgraph,group_edge_attrs=['embedding', 'relevance_label'])
        else:
            data = _from_multigraph_networkx(nxgraph,group_edge_attrs=['embedding'])
        if not is_directed(nxgraph):
            data.edge_index,data.edge_attr=remove_self_loops(data.edge_index,data.edge_attr)
        data = LineGraph(force_directed=True)(data)
        return data

    def update(self, original_graph: Optional[MultiDiGraph], new_graph: Optional[MultiDiGraph], relevance_label: bool= False) -> Optional[MultiDiGraph]:
        '''
        used to update the graph with new triples
        Arguments:
            original_graph: networkx graph instance,
            new_graph: networkx graph instance,
            relevance_label: boolean, whether to utilise relevance labels as attributes
        '''
        if original_graph is None:
            return new_graph
        if new_graph is None:
            return original_graph
        new_graph = compose(original_graph,new_graph)
        edge_attributes, edge_labels = self.embed(new_graph)
        new_graph = self._update_attributes(new_graph, edge_attributes, edge_labels, relevance_label)
        return new_graph

    def forward(self,nxgraph: Optional[MultiDiGraph], relevance_label=True) -> Optional[Data]:
        '''
        arguments:
            nxgraph: a networkx (knowledge) graph,
            relevance_label: boolean, whether to utilise relevance labels
        returns:
            data, edge_index (torch_geometric format)
        Changes the networkx graph into a line graph.

        use the edge_index to reverse the line graph transform (for checking the text on the original triple).
        '''
        if nxgraph is None or nxgraph.number_of_edges() == 0:
            return None
        edge_attributes, edge_labels = self.embed(nxgraph)
        nxgraph = self._update_attributes(nxgraph, edge_attributes, edge_labels, relevance_label = relevance_label)
        return self.transform(nxgraph, relevance_label = relevance_label)

    def add_query(self, graph_data: Optional[Data], query, relevance_label=True) -> Optional[Data]:
        """
        Adds the query to the embedding... also seperates out the label from the attributes
        """
        if graph_data is None:
            return None
        query_embedding = self.lm_embedder(query)
        if relevance_label:
            graph_data.y = graph_data.x[:,-1]
            graph_data.x = graph_data.x[:,:-1]
        graph_data.x = cat([graph_data.x,query_embedding.expand((graph_data.x.size(0),-1))], dim=1) ## concats on query embedding
        return graph_data

def _from_multigraph_networkx(G, group_node_attrs: Optional[Union[List[str], all]] = None,
                  group_edge_attrs: Optional[Union[List[str], all]] = None):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    modified version of the from_networkx function in torch_geometric.utils.convert designed to work with multigraphs

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.
    """
    import networkx as nx

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = LongTensor(list(G.edges)).t().contiguous()

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in data.items():
        try:
            data[key] = tensor(value)
        except ValueError:
            pass
    data['edge_index'] = edge_index.view(3, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data

if __name__ == '__main__':
    from networkx import MultiDiGraph, Graph
    nxgraph = MultiDiGraph()
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
    newgraph = MultiDiGraph()
    newgraph.add_nodes_from(["hi","salut"])
    newgraph.add_edge("hi","salut",label="is_same")
    newgraph = gt.update(nxgraph,newgraph)
    graph, edge_index = gt.transform(newgraph,relevance_label=False)
    graph = gt.add_query(graph, "help")
    print(graph)
    print(graph.x)