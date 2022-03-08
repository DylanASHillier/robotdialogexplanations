from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch import mean, stack
from torch.optim import Adam

class Triples2TextSystem(LightningModule):
    def __init__(self,base_model="t5",lr=1e-5):
        super(Triples2TextSystem,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        self.lr = lr
        self.save_hyperparameters()

    def setup_train(self):
        self.model.train()

    def forward(self,input):
        '''
        input textual triple of shape [batchsize, maximum_length]
        '''
        input = self.tokenizer(input)
        outputs = self.model(input)[:,:-len(input)]
        return outputs

    def training_step(self,batch,batch_idx):
        return self.update_step(batch, "train")

    def update_step(self,batch,log_string):
        batched_triples_ids, batched_triple_mask, batched_target_ids, batched_target_mask = batch      
        outputs = self.model(batched_triples_ids, labels=batched_target_ids)
        loss = outputs[0]
        self.log(f"{log_string}_loss",loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        return self.update_step(batch, "validation")

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)
        