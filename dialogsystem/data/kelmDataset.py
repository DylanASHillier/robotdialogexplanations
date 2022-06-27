from torch import tensor
from torch.utils.data import Dataset,DataLoader
import json
## Outdated

class KELMDataset(Dataset):
    def __init__(self,jsons) -> None:
        self.jsons = jsons

    def __getitem__(self, index):
        sample = json.loads(self.jsons[index])
        return sample["serialized_triples"],sample["gen_sentence"]

    def __len__(self):
        return len(self.jsons)

class Kelm_dataloader(DataLoader):
    def __init__(self, tokenizer, *args, **kwargs):
        super(Kelm_dataloader,self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.collate_fn = self.kelm_collate

    def kelm_collate(self,samples):
        tokenized_triples = self.tokenizer(list(map(lambda x:x[0], samples)),truncation=True,padding=True,return_tensors='pt')
        tokenized_labels = self.tokenizer(list(map(lambda x:x[1], samples)),truncation=True,padding=True,return_tensors='pt')
        return tokenized_triples["input_ids"],tokenized_triples["attention_mask"],tokenized_labels["attention_mask"],tokenized_labels["attention_mask"]