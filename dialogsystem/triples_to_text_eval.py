# evaluate trained conv_qa model on coqa dataset
from models.triples2text import PretrainedTriples2TextSystem
from models.triples2text import Triples2TextSystem
import argparse
from datasets import load_dataset
from torchmetrics import SQuAD, BLEUScore
from torchmetrics.functional import bleu_score
from tqdm import tqdm
from numpy import random

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='dialogsystem/trained_models/t2t/t2ttrained',help='Where model is loaded from')

    # def preprocess_function(samples):
    #     inputs = [prefix + sample for sample in samples[source]]
    #     targets = [sample for sample in samples[target]]
    #     model_inputs = tokenizer(inputs, max_length=256, truncation=True)
    #     with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(targets, max_length=256, truncation=True)
    #     model_inputs["labels"]=labels["input_ids"]
    #     return model_inputs

if __name__ == '__main__':
    F1 = SQuAD()
    bleu = 0
    raw_F1 = SQuAD()
    raw_bleu = 0
    num_samples = 1000
    args = parser.parse_args()
    # model = Triples2TextSystem(args.model_path)
    model = PretrainedTriples2TextSystem() # choose 1 of the 2 models
    
    dataset = load_dataset("kelm")
    random.seed(10)
    ds_test = random.choice(dataset["test"], size=num_samples)
    mock_idx = 0
    for sample in tqdm(ds_test):
        prediction = model([sample["triple"]])[0]
        # print(f"generated:{prediction}\n target:{sample['sentence']}\n source: {sample['triple']}\n\n")
        squad_prediction = {
            "prediction_text": prediction,
            "id": mock_idx
        }
        squad_source = {
            "prediction_text": sample["triple"],
            "id": mock_idx
        }
           
        squad_target = {
            "answers": 
                {
                    "answer_start": [mock_idx],
                    "text": [sample["sentence"]]
                }
            ,
            "id": mock_idx
        }
        F1.update(squad_prediction, squad_target)
        bleu+= bleu_score(prediction, [sample["sentence"]])
        raw_F1.update(squad_source, squad_target)
        raw_bleu += bleu_score(sample["triple"], [sample["sentence"]])
        # raw_metrics.update(squad_source, squad_target)
        # metrics.update(squad_prediction, squad_target)
        mock_idx += 1
    print(f"raw_metrics: F1: {raw_F1.compute()}, bleu: {raw_bleu/num_samples}")
    print(f"metrics: F1: {F1.compute()}, bleu: {bleu/num_samples}")
