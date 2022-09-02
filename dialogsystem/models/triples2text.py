from pytorch_lightning import LightningModule
from transformers import AutoTokenizer, T5ForConditionalGeneration, TextGenerationPipeline, AutoModelForCausalLM
from torch.optim import Adam
from torch import no_grad
import warnings

generation_params={
"temperature":0.5,
"do_sample":True,
"top_p":0.4,
"repetition_penalty":1.5,
"length_penalty":0.01,
"max_new_tokens": 50
}

class Triples2TextSystem(LightningModule):
    '''
    source = "triple"
    prefix = "translate triples to text: "
    '''
    def __init__(self,model_checkpoint="t5-small",lr=1e-5):
        super(Triples2TextSystem,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
        self.lr = lr
        self.prefix = "translate triples to text: "
        self.save_hyperparameters()

    def setup_train(self):
        self.model.train()

    def forward(self,input):
        '''
        input textual triple of shape [batchsize, maximum_length]
        '''
        with no_grad():
            input = [self.prefix + sample for sample in input]
            input = self.tokenizer(input, truncation=True, padding=True, return_tensors="pt").input_ids
            outputs = self.model.generate(input, **generation_params)
            outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
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

class PretrainedTriples2TextSystem(LightningModule):
    '''
    Alternate version of Triples2TextSystem that uses a pretrained causal language model
    (e.g. GPT-Neo) instead of T5.
    '''
    def __init__(self,model_checkpoint="EleutherAI/gpt-neo-1.3B",lr=1e-5):
        super(PretrainedTriples2TextSystem,self).__init__()
        model_string = model_checkpoint
        self.model = AutoModelForCausalLM.from_pretrained(model_string)
        self.tokenizer = AutoTokenizer.from_pretrained(model_string)
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer, device=-1)
        self.prefix = "translate triples to text:(example) man, is, cool; man, is cook -> the man is a cool cook\n (example) jack, is quick at, walking; jack, is, tall -> jack is both tall and a fast walker \n(actual) "
        self.save_hyperparameters()

    def setup_train(self):
        warnings.warn("setup_train is deprecated, this is a pretrained model", DeprecationWarning)
        
    def forward(self,input):
        '''
        input textual triple of shape [batchsize, maximum_length]
        '''
        output_strings = [self.pipeline(
        f"{self.prefix} {string} ->", 
        return_full_text=False,
        **generation_params,
        )[0]["generated_text"].split("\n")[0] for string in input]
        return output_strings

    def validation_step(self, batch, batch_idx):
        warnings.warn("validation_step is deprecated, this is a pretrained model", DeprecationWarning)

    def configure_optimizers(self):
        return Adam(self.parameters(), self.lr)

if __name__ == '__main__':
    t2t = Triples2TextSystem("./dialogsystem/trained_models/t2t/t2ttrained")
    alt_t2t = PretrainedTriples2TextSystem()
    print(t2t(["robot,holds,box;box,is,red;robot,is,tiago","robot,in,shower;robot,is,singing"]))
    print(alt_t2t(["tiago,holds,box; box,is,red; tiago,is,a robot","robot,in,shower;robot,is,singing"]))
        