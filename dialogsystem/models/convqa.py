from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, BertForQuestionAnswering, T5ForConditionalGeneration
from torch import mean, stack, no_grad
from torch.optim import Adam

class ConvQASystem(LightningModule):
    '''
    NB this is never actually trained by me, we just use a lightning wrapper for convenience
    '''
    def __init__(self,base_model="bert-base-uncased",lr=1e-5):
        super(ConvQASystem,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = BertForQuestionAnswering.from_pretrained(base_model)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self,input):
        '''
        input question and background
        '''
        with no_grad():
            input = self.tokenizer(input, truncation=True, padding=True, return_tensors="pt").input_ids
            outputs = self.model.generate(input)
            outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return outputs

    def load_from_hf_checkpoint(self, checkpoint_path):
        # self.model = BertForQuestionAnswering.from_pretrained(checkpoint_path)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        return self

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)
        