import os
from typing import List, Tuple, Union
from torch_geometric.data import Dataset
from networkx.readwrite import gpickle
from torch import save, load
from models.GraphEmbedder import GraphTransformer
from data import qtext
from re import search
from tqdm import tqdm
from zipfile import ZipFile

def get_index(graphname):
    match = search('\d+',graphname)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"{graphname} does not contain a number")

class GraphTrainDataset(Dataset):
    def __init__(self,root="datasets/processed_ds", raw_filepath='datasets', datasets = ['ropes','coqa','squad'],graph_transformation_LM='t5-small',transform=None, pre_transform=None, pre_filter=None) -> None:
        """
        graph_transformation_LM
        To fit onto disk I call `find . -type f -name "data_*.pt" -execdir zip -m '{}.zip' '{}' \; `
        """
        self.graphembedder=GraphTransformer(graph_transformation_LM)  
        self.init_jsons = []
        self.raw_filepath = raw_filepath
        for ds in datasets:
            if ds=='ropes':
                text_ds = qtext.QtextRopes("train")
            elif ds=='squad':
                text_ds = qtext.QtextSQUAD("train")
            elif ds=='coqa':
                text_ds = qtext.QtextCoQA("train")
            graph_files = os.listdir(raw_filepath+'/'+ds)
            graph_files.remove("README.md")
            self.init_jsons += [(text_ds[get_index(e)][0],raw_filepath+'/'+ds+'/'+e)  for i,e in enumerate(graph_files)]
        super(GraphTrainDataset, self).__init__(root,transform,pre_transform,pre_filter)


    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"data_{i}.pt.zip" for i,_ in enumerate(self.init_jsons)]

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property          
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self.init_jsons

    def process(self):
        idx = 0
        for query, graph in tqdm(self.init_jsons):
            sample = gpickle.read_gpickle(graph)
            graph, _ = self.graphembedder(sample)
            data = self.graphembedder.add_query(graph,query)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def get(self, idx):
        with ZipFile(os.path.join(self.processed_dir, f'data_{idx}.pt.zip')) as zf:
            for file in zf.namelist():
                with zf.open(file) as f:
                    graph = load(f)
                    x = graph.x[:,:graph.x.size(1)-1]
                    y = graph.x[:,-1]
                    graph.x = x
                    graph.y = y
                    return graph

    def len(self):
        return len(self.init_jsons)

if __name__ == '__main__':
    gtds = GraphTrainDataset()
    print(gtds[0])