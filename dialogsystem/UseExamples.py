from kb_retrieval.data_extraction import DataExtractor
from kb_retrieval.graph_construction import GraphConstructor
from tqdm import tqdm

if __name__ == '__main__':
    ### Example Script for building graphs
    gc = GraphConstructor()
    de = DataExtractor()
    entities = de.get_entities_from_falcon2("what is the value of my health. Where are the ducks. What is the point of getting sober in Toronto?")
    print(len(entities))
    for entity in tqdm(entities):
        triples = de.triples_from_query(entity["id"],entity["label"])
        for triple in triples:
            gc.add_from_triple(triple)
    print(gc.build_graph())