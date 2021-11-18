from data.kelmDataset import KELMDataset
from data.dataimport import DataImporter
from data.kelmDataset import KELMDataset

di = DataImporter()
data = di.load_kelm_data()
train,test,val = di.split_data(data,0.9,0.5,0.5)
train_dataset,test_dataset,val_dataset = KELMDataset(train),KELMDataset(test),KELMDataset(val)
[print(train_dataset[i]) for i in range(10)]
