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

    def setup_train(self):
        self.model.train()

    def forward(self,input):
        input = self.tokenizer(input)
        outputs = self.model(input)
        return 

    def training_step(self,batch,batch_idx):
        input_idss = [it["input_ids"] for it in batch[0]]
        attention_masks = [it["attention_mask"] for it in batch[0]]

        return self.update_step(input_idss,attention_masks,"train")

    def update_step(self,input_idss,attention_masks,labels,log_string):        
        input_idxs = input_idxs + [input_idss[idx]]
        outputs = self.model(cat(input_idxs), labels=target_input_idss[idx])
        input_idxs = input_idxs + target_input_idss[idx] ## Add target turn to context for next turn
        loss = outputs[0]
        self.log(f"{log_string}_loss",loss.item())
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)
        