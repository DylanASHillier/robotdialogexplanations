from datasets import load_dataset
from torch.utils.data import Dataset
from numpy import random

class QtextRopes(Dataset):
    def __init__(self, split, base_dataset='ropes'):
        '''
        Arguments:
            base_dataset: which dataset to use
            split: which split to use for the dataset
        '''
        self.dataset = load_dataset(base_dataset,split=split)

    def __getitem__(self, index):
        item = self.dataset[index]
        question = item['question']
        conv_context = item['situation']
        background = item['background']
        return question, conv_context, background

    def __len__(self):
        return len(self.dataset)

class QtextCoQA(Dataset):
    def __init__(self, split, base_dataset='coqa'):
        self.dataset = load_dataset(base_dataset,split=split)
        self.rng = random.default_rng(10)

    def __getitem__(self, index):
        item = self.dataset[index]
        rand_index = self.rng.integers(0,len(item['questions']))
        question = item['questions'][rand_index]
        qapairs = zip(item['questions'][:rand_index],item['answers']['input_text'][:rand_index])
        conv_context = ' \n '.join(list(map(lambda x: x[0]+' \n '+x[1],qapairs )))
        background = item['story']
        return question, conv_context, background

    def __len__(self):
        return len(self.dataset)

class QtextSQUAD(Dataset):
    def __init__(self, split, base_dataset='squad'):
        self.dataset = load_dataset(base_dataset,split=split)

    def __getitem__(self, index):
        item = self.dataset[index]
        question = item['question']
        background = item['context']
        conv_context = ""
        return question, conv_context, background

if __name__=='__main__':
    ds = QtextRopes('train')
    print(ds)
    print(ds[0])
    ds = QtextCoQA('train')
    print(ds)
    print(ds[0])
    ds = QtextSQUAD('train')
    print(ds)
    print(ds[0])