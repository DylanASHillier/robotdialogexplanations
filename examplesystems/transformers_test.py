from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, T5ForConditionalGeneration
from flask import Flask, request, json

app = Flask(__name__)

## hugging face generation models: 
# # tried "facebook/blenderbot-1B-distill" , 'facebook/bart-base', but ERROR
# "bigscience/bloom-560m" , ctrl-small or ctrl, 
# EleutherAI/gpt-neo-125M : good effort to combine triplets 
# gpt-neo-1.3B
# RUCAIBox/mvp (returns gibberish, even tried in format "Describe the data: robot|likes|apples"), google/pegasus (which one to choose? documentation is not very clear)
# "gpt2" # returns code ? not NL sentence? gptj too large
model_string = "EleutherAI/gpt-neo-1.3B"  #BEST

#robot, is, big; robot, like, apples
#robot, visited, waypoint; waypoint, is, kitchen

model = AutoModelForCausalLM.from_pretrained(model_string)
#model = T5ForConditionalGeneration.from_pretrained("dialogsystem/trained_models/t2t/t2ttrained")
tokenizer = AutoTokenizer.from_pretrained(model_string)
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)
generation_params={
"temperature":0.05,
"do_sample":True,
"top_p":0.4,
"repetition_penalty":1.5,
"length_penalty":0.01,
"max_length":200, #or max_new_tokens?
# could have top_k = 50 activated too
 
}

#@app.route('/', methods=['POST'])
# def infer():
#     '''
#     Receives text input from post request and runs infernece on it.
#     '''
#     text = request.form['text']
#     out = pipeline(
#     text, 
#     return_full_text=False,
#     generation_kwargs=generation_params
#     )
#     out = out[0]["generated_text"]
#     return out

if __name__=='__main__':
    while True:
        input_triple = input("Enter a sentence: ")
        input_string = f"translate triples to text: tiago, is, robot; robot, holds, box; box, is, red -> tiago holds a red box\n robot,in,shower;robot,is,singing -> a robot is singing in the shower\n robot, at, wp2 ; robot, is, blue -> a blue robot is at wp2\n robot, in, kitchen; kitchen, has, table -> the robot is in the kitchen with a table {input_triple}->"
        out = pipeline(
        input_string, 
        return_full_text=False,
        **generation_params,
        )
        out = out[0]["generated_text"]
        print(out)