# evaluate trained conv_qa model on coqa dataset
import sagemaker
from models.convqa import ConvQASystem
import argparse
from datasets import load_dataset
from torchmetrics import SQuAD
from tqdm import tqdm
in_metrics = SQuAD()
out_metrics = SQuAD()

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='dialogsystem/trained_models/convqa',help='Where model is loaded from')

in_domain = ["mctest", "gutenberg", "race", "cnn", "wikipedia"]
out_domain = ["reddit", "science"]

background_pref = "background: "
context_pref = "context: "
question_pref = "question: "
target_pref = "answer: "
prefix = "Answer the Question: "

# def preprocess_function(samples):
#     rng = random.default_rng(10)
#     inputs = []
#     targets = []
#     for idx in range(len(samples["questions"])):
#         rand_index = rng.integers(0,len(samples['questions'][idx]))
#         question = samples['questions'][idx][rand_index]
#         answer = samples['answers'][idx]['input_text'][rand_index]
#         qapairs = zip(samples['questions'][idx][:rand_index],samples['answers'][idx]['input_text'][:rand_index])
#         conv_context = ' \n '.join(list(map(lambda x: x[0]+' -> ' +x[1],qapairs )))
#         background = samples['story'][idx]
#         inputs.append(f"{background_pref} {background} \n {context_pref} {conv_context} \n {question_pref} {question}")
#         targets.append(f"{target_pref} {answer}")
    
#     model_inputs = tokenizer(inputs, max_length=256, truncation=True)
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(targets, max_length=256, truncation=True)
#     model_inputs["labels"]=labels["input_ids"]
#     return model_inputs


if __name__ == '__main__':
    args = parser.parse_args()
    print("ready")
    model = ConvQASystem(args.model_path)
    
    dataset = load_dataset("coqa")
    ds_test = dataset["validation"]
    mock_idx = 0
    for sample in tqdm(ds_test):
        inputs = []
        targets = []
        for index in range(len(sample['questions'])):
            domain = sample["source"]
            question = sample['questions'][index]
            answer = sample['answers']['input_text'][index]
            qapairs = zip(sample['questions'][:index],sample['answers']['input_text'][:index])
            conv_context = ' \n '.join(list(map(lambda x: x[0]+' -> ' +x[1],qapairs )))
            background = sample['story']
            input = f"{background_pref} {background} \n {context_pref} {conv_context} \n {question_pref} {question}"
            inputs.append(input)

            target = f"{target_pref} {answer}"
            targets.append(target)
        predictions = model(inputs)
        for idx in range(len(predictions)):
            squad_prediction = {
                "prediction_text": predictions[idx],
                "id": mock_idx
            }
           
            squad_target = {
                "answers": 
                    {
                        "answer_start": [mock_idx],
                        "text": [targets[idx]]
                    }
                ,
                "id": mock_idx
            }
            if input==target:
                print("heya")
            if domain in in_domain:
                in_metrics.update(squad_prediction, squad_target)
            elif domain in out_domain:
                out_metrics.update(squad_prediction, squad_target)
            else:
                print(f"Domain {domain} not found")
            mock_idx += 1
    print(f"In domain: {in_metrics.compute()}")
    print(f"Out domain: {out_metrics.compute()}")

            

