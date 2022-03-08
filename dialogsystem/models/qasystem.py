from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch import mean, stack, cat
from torch.optim import Adam
from models.kgqueryextract import KGQueryMPNN
from models.GraphEmbedder import LMEmbedder,GraphDataEmbedder
from models.triples2text import Triples2TextSystem
from transformers import T5EncoderModel, T5Tokenizer

class QASystem(LightningModule):
    def __init__(self,t2t_weights,lm_embedding_size,base_model='t5',lr=1e-5):
        super(QASystem,self).__init__()
        self.triples2text = Triples2TextSystem()
        self.triples2text.load_state_dict(t2t_weights)
        t5encoder = T5EncoderModel.from_pretrained(base_model)
        t5tokenizer = T5Tokenizer.from_pretrained(base_model)
        self.t5encoder=LMEmbedder(t5encoder,t5tokenizer)
        for param in self.t5encoder.parameters():
            param.requires_grad=False
        for param in self.triples2text.parameters():
            param.requires_grad=False
        self.graphembedder = GraphDataEmbedder(lm_embedding_size,self.t5encoder)
        self.kggnn = KGQueryMPNN(3,20,2)
        self.lr = lr

    def setup_train(self):
        self.model.train()

    def forward(self,query,graph):
        '''
        Arguments:
            query: string
            graph: the graph overwhich the question is run
        '''
        embedded_graph = self.graphembedder(graph,query)
        transformed_graph, edge_index = self.graphtransformer(embedded_graph)
        activated_edges = self.kggnn(transformed_graph)
        triples = []
        for edge in activated_edges:
            triples.append(edge_index[edge])
        t2toutput = self.triples2text(cat(triples),query)
        return t2toutput

    def training_step(self,batch,batch_idx):
        batched_triples, batched_targets = batch
        return self.update_step(batched_triples, batched_targets, "train")

    def update_step(self,src_encodings,target_encodings,log_string):
        '''
        Calculates the loss. For this model loss is equivalent to:
        confidence(triples)*encoding_loss
        '''     
        outputs = self.model(src_encodings, labels=target_encodings)
        loss = outputs[0]
        self.log(f"{log_string}_loss",loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        batched_triples, batched_targets = batch
        return super().validation_step(batched_triples, batched_targets, "test")

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)
        