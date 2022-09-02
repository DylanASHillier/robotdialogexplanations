from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
model_string = "EleutherAI/gpt-neo-1.3B"  #BEST
# model_string = "facebook/opt-125m"

model = AutoModelForCausalLM.from_pretrained(model_string)
tokenizer = AutoTokenizer.from_pretrained(model_string)
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)
generation_params={
"temperature":0.5,
"do_sample":True,
"top_p":0.4,
"repetition_penalty":1.5,
"length_penalty":0.01,
"max_new_tokens": 50
}

if __name__=='__main__':
    while True:
        input_triple = input("Enter a sentence: ")
        input_string = f"translate triples to text: tiago, is, robot; robot, holds, box; box, is, red -> tiago holds a red box\n robot,in,shower;robot,is,singing -> a robot is singing in the shower\n robot, at, wp2 ; robot, is, blue -> a blue robot is at wp2\n robot,is in, kitchen; kitchen, has, table -> a robot is in the kitchen with a table \n box, is, red; box, is in, kitchen; robot, holds, box; robot, is, blue -> A blue robot is holding a red box in the kitchen{input_triple}->"
        out = pipeline(
        input_string, 
        return_full_text=False,
        **generation_params,
        )
        out = out[0]["generated_text"]
        print(out)