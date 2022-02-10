from torch.utils.data import Dataset
import json

class KELMDataset(Dataset):
    def __init__(self,jsons) -> None:
        self.jsons = jsons

    def __getitem__(self, index):
        triple = json.loads(self.jsons[index])
        return triple

    def __len__(self):
        return len(self.triples)