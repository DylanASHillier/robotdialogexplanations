from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from data.dataimport import DataImporter
from data.kelmDataset import KELMDataset,Kelm_dataloader
from data.qtext import QtextSQUAD
import argparse
from data.graphDataset import GraphTrainDataset
from models.GraphEmbedder import LMEmbedder
from models.triples2text import Triples2TextSystem
from models.kgqueryextract import LightningKGQueryMPNN
from models.GraphEmbedder import LitAutoEncoder
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch_geometric
from torch.utils.data import Dataset

#parse arguments for model training.
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system_type',type=str, default='t2t', help='system being trained, choose from \'t2t\',\'gnn\'')
    parser.add_argument('--base_model', type=str, default='t5-small', help='pretrained transformers model to be used')
    parser.add_argument('--num_epoch', type=int, default=5, help='number of maximum epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize')
    parser.add_argument('--pretrained_model_loc', type=str, default=None, help='folder containing any pretrained model weight')
    parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate used by the optimizer')
    parser.add_argument('--model_output_path', type=str, default='model.ckpt',help='Where model is stored after training')
    parser.add_argument('--debug', default=False,action='store_true', help='use small training set and disable logging for quickfire testing')
    parser.add_argument('--num_gpus', type=int, default=4,help='number of gpus to be used')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_arguments()    
    if args.debug:
        logger = None
    else:
        logger=NeptuneLogger(api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZTZiNDUxYi03ZDNiLTQ3N2EtYjQwMC0wZjA0NTJiNTgwZDQifQ==",
                        project="dylanslavinhillier/robodialog")
    di = DataImporter()
    if args.system_type=='t2t':
        model = Triples2TextSystem(args.base_model,args.lr)
        model.setup_train()  
        print("model setup")
        data = di.load_kelm_data("/data/kelm_generated_corpus.jsonl")
        train,test,val = di.split_data(data,0.9,0.05,0.05)
        train_dataset,test_dataset,val_dataset = KELMDataset(train),KELMDataset(test),KELMDataset(val)
        print("datasets setup")
        train_dl = Kelm_dataloader(model.tokenizer,train_dataset,args.batch_size)
        val_dl =  Kelm_dataloader(model.tokenizer,val_dataset,args.batch_size)
    elif args.system_type == 'gnn':
        train_dataset = GraphTrainDataset("datasets")
        print("dataset setup")
        model = LightningKGQueryMPNN(1024)
        print("model setup")
        train_dl = torch_geometric.data.DataLoader(train_dataset,args.batch_size,num_workers=4)
        val_dl = None
    elif args.system_type == 'autoencoder':
        lmodel = T5ForConditionalGeneration.from_pretrained(args.base_model).encoder
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = LitAutoEncoder(512,25)
        lmembedder = LMEmbedder(lmodel,tokenizer)
        class text_ds(Dataset):
            def __init__(self, texts) -> None:
                super().__init__()
                self.texts = texts
            
            def __len__(self) -> int:
                return len(self.texts)

            def __getitem__(self, idx):
                return lmembedder(self.texts[idx])
        
        train_dataset = text_ds([item[0] for item in QtextSQUAD("train")])
        train_dl = DataLoader(train_dataset)
        val_dataset = text_ds([item[0] for item in QtextSQUAD("validation")])
        val_dl = DataLoader(val_dataset)
    if args.num_gpus>0:
        trainer = Trainer(logger=logger,log_every_n_steps=10,max_epochs=args.num_epoch,gpus=args.num_gpus,enable_checkpointing=False,strategy='ddp')
    else:
        trainer = Trainer(logger=logger,log_every_n_steps=10,max_epochs=args.num_epoch,gpus=args.num_gpus,enable_checkpointing=False)
    print("beginning train")
    trainer.fit(model,train_dl,val_dl)
    print("train finished, saving mdoel")
    trainer.save_checkpoint(args.model_output_path)
    print("finished")