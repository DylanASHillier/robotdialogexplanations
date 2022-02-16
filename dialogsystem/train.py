from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from data.kelmDataset import KELMDataset
from data.dataimport import DataImporter
from data.kelmDataset import KELMDataset
import argparse
from data.qtext import QtextCoQA, QtextRopes, QtextSQUAD
from models.triples2text import Triples2TextSystem

#parse arguments for model training.
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system_type',type=str, default='t2t', help='system being trained, choose from \'t2t\',\'gnn\'')
    parser.add_argument('--base_model', type=str, default='t5', help='pretrained transformers model to be used')
    parser.add_argument('--num_epoch', type=int, default=5, help='number of maximum epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize')
    parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate used by the optimizer')
    parser.add_argument('--model_output_path', type=str, default='dialogsystem/trained_models/model.ckpt',help='Where model is stored after training')
    parser.add_argument('--debug', type=bool, default=False, help='use small training set and disable logging for quickfire testing')
    parser.add_argument('--num_gpus', type=int, default=1,help='number of gpus to be used')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_arguments()
    if args.debug:
        logger = None
    else:
        logger=NeptuneLogger(api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZTZiNDUxYi03ZDNiLTQ3N2EtYjQwMC0wZjA0NTJiNTgwZDQifQ==",
                        project="dylanslavinhillier/robotdialog")
    di = DataImporter()
    if args.system_type=='t2t':
        data = di.load_kelm_data()
        train,test,val = di.split_data(data,0.9,0.5,0.5)
        train_dataset,test_dataset,val_dataset = KELMDataset(train),KELMDataset(test),KELMDataset(val)
        print("datasets setup")
        model = Triples2TextSystem(args.base_model,args.lr)
        model.setup_train()  
        print("model setup")
    elif args.system_type == 'gnn':
        train_datasets = [QtextSQUAD('train'),QtextCoQA('train'),QtextRopes('train')]
        valid_datasets = [QtextSQUAD('validation'),QtextCoQA('validation'),QtextRopes('validation')]
    
    if args.num_gpus>0:
        trainer = Trainer(logger=logger,log_every_n_steps=10,max_epochs=args.num_epoch,gpus=args.num_gpus,enable_checkpointing=False,strategy='ddp')
    else:
        trainer = Trainer(logger=logger,log_every_n_steps=10,max_epochs=args.num_epoch,gpus=args.num_gpus,enable_checkpointing=False)
    trainer.fit(model,DataLoader(train_dataset,args.batch_size,num_workers=4),DataLoader(val_dataset,args.batch_size,num_workers=4))

    trainer.save_checkpoint(args.model_output_path)