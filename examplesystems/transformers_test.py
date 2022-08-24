from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from flask import Flask, request, json

app = Flask(__name__)

## hugging face generation models
model_string = "t5-small"
model_string = "facebook/opt-13b"

model = AutoModelForCausalLM.from_pretrained(model_string)
tokenizer = AutoTokenizer.from_pretrained(model_string)
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)

generation_params={
"temperature":0.8,
"do_sample":True,
"top_p":0.95,
"repetition_penalty":1.5,
"length_penalty":0.01,
"max_tokens":500 #or max_length?
# could have top_k = 50 activated too
"num_return_sequences":3
}

@app.route('/', methods=['POST'])
def infer():
    '''
    Receives text input from post request and runs infernece on it.
    '''
    text = request.form['text']
    out = pipeline(
    text, 
    return_full_text=False,
    generation_kwargs=generation_params
    )
    out = out[0]["generated_text"]
    return out

if __name__=='__main__':
    while True:
        text = input("Enter a sentence: ")
        out = pipeline(
        text, 
        return_full_text=False,
        generation_kwargs=generation_params
        )
        out = out[0]["generated_text"]
        print(out)