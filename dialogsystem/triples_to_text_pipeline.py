from datasets import load_dataset
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

def parse_arguments():
    '''
    parse arguments for 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--model_output_path', type=str, default='models/trained_models/t2t',help='Where model is stored after training')
    parser.add_argument('--model_name', type=str, default='t5-base', help='model name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    tokenizer=T5Tokenizer.from_pretrained(args.model_name)
    model=T5ForConditionalGeneration.from_pretrained(args.model_name)

    source = "triple"
    target = "sentence"
    prefix = "translate triples to text: "

    def preprocess_function(samples):
        inputs = [prefix + sample for sample in samples[source]]
        targets = [sample for sample in samples[target]]
        model_inputs = tokenizer(inputs, max_length=256, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=256, truncation=True)
        model_inputs["labels"]=labels["input_ids"]
        return model_inputs
    ds = load_dataset("kelm")
    ds_train = ds["train"].map(preprocess_function, batched=True)
    ds_val = ds["validation"].map(preprocess_function, batched=True)


    training_args = Seq2SeqTrainingArguments(
        output_dir="./dialogsystem/trained_models/",
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_total_limit=3,
        max_grad_norm=1.0,
    )

    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )


    Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=data_collator
    ).train()

    tokenizer.save_pretrained(args.model_output_path)
    model.save_pretrained(args.model_output_path)