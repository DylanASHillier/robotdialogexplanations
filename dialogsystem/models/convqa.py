from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch import no_grad
from torch.optim import Adam

class ConvQASystem(LightningModule):
    '''
    NB this is never actually trained using this class, we just use a lightning wrapper for convenience
    Arguments:
        checkpoint_path: path to the checkpoint to load (hugging face checkpoint)
    '''
    def __init__(self,checkpoint_path):
        super(ConvQASystem,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
        self.lr = 1e-5
        self.save_hyperparameters()

    def forward(self,input):
        '''
        input question and background
        '''
        with no_grad():
            input = self.tokenizer(input, truncation=True, padding=True, return_tensors="pt").input_ids
            outputs = self.model.generate(input)
            outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            if len(outputs) == 1:
                return outputs[0]
            return outputs

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)

if __name__ == "__main__":
    model = ConvQASystem("dialogsystem/trained_models/convqa")
    print(model("What is the weather in New York?"))
    print(model(["What is the weather in New York?","What is the weather in London?"]))
