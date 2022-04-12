from kb_retrieval.data_extraction import DataExtractor
from networkx.readwrite import gpickle

de = DataExtractor()
# relations, entities, graph = de.triples_etc_from_csv("datasets/KGs/conceptnetassertions.csv",'rdflib')
relations, entities, graph = de.triples_etc_from_conceptnet_csv("datasets/KGs/conceptnetassertions.csv")
gpickle.write_gpickle(graph,f"datasets/KGs/conceptnet.json")