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
        input_string = f"translate triples to text: man, is, cool; man, is cook -> the man is a cool cook\n jack, is quick at, walking; jack, is, tall -> jack is both tall and a fast walker \n {input_triple}->"
        out = pipeline(
        input_string, 
        return_full_text=False,
        **generation_params,
        )
        out = out[0]["generated_text"]
        print(out)