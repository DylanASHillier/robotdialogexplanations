import json
from tqdm import tqdm

class DataImporter():

    def split_data(self,data,train_portion,test_portion,val_portion):
        '''
        Code splits data according to the train, test, and validation portions
        Arguments:
            data: list, data that can be indexed over
            train_portion: float, portion of data used for training
            test_portion: float, portion of data used for evaluating
            val_portion: float, portion of data used for validating
        Example:
            split_data([0,..,100],0.6,0.1,0.3) would return datasets [0,...,~60],[~61,...,~70],[~71,...100]
        '''
        train_last = int(train_portion*len(data))
        test_last = int(train_last + test_portion*len(data))
        val_last = int(test_last + val_portion*len(data))
        return data[0:train_last],data[train_last:test_last],data[test_last:val_last]

    def load_kelm_data(self, path="datasets/KELMCorpus/kelm_generated_corpus.jsonl"):
        '''
        Code finds and prepares data into a list for KELMCorpus. These are then loaded in by the dataset class
        Arguments:
            path: string, location of data
        returns:
            list<json_file>
        '''
        with open(path, 'r') as f:
            entity_jsons = list(f)
        return entity_jsons

if __name__ == '__main__':
    di = DataImporter()
    ent = di.load_kelm_data()
    print(len(ent))
    print(ent[0])