from torch.utils.data import Dataset
import json
from networkx import MultiDiGraph

class gqa(Dataset):
    def __init__(self, split):
        '''
        Arguments:
            base_dataset: which dataset to use
            split: which split to use for the dataset
        '''
        ## load graphs from json
        with open(f'datasets/scene_graphs/{split}_sceneGraphs.json', 'r') as f:
            self.graphs = json.load(f)
        with open(f'datasets/scene_graphs/{split}_balanced_questions.json', 'r') as f:
            self.questions = list(json.load(f).values())

    def _convert_to_nx(self, graph):
        '''
        Converts a graph to a networkx graph
        '''
        nx_graph = MultiDiGraph()
        for node, attr in graph["objects"].items():
            nx_graph.add_edge(node, attr["name"], label = 'is')
            nx_graph.add_edge(attr["name"], node, label = 'is')
            for pred in attr["attributes"]:
                nx_graph.add_edge(node, pred, label = "is")
                nx_graph.add_edge(node, pred, label = 'is')
            for rel in attr["relations"]:
                nx_graph.add_edge(node, rel["object"], label = rel["name"])
        return nx_graph

    def __getitem__(self, index):
        q_json = self.questions[index]
        question = q_json["question"]
        graph_id = q_json["imageId"]
        graph = self.graphs[graph_id]
        graph = self._convert_to_nx(graph)
        answer = q_json["fullAnswer"]
        return question, answer, graph

    def __len__(self):
        return len(self.questions)

if __name__ == '__main__':
    ds = gqa('train')
    print(len(ds))
    print(ds[0])