from html import entities
from requests import get, post
from SPARQLWrapper import SPARQLWrapper2
import pandas as pd
import json

class DataExtractor():
    '''
    Class containing methods to extract networkx graphs for use as data sources:
    Arguments:
        k: the k-hop neighbourhood of entities to use
    Methods:
        de.triples_from_query(entity,kb): uses sparql queries to construct the graph
    '''
    def __init__(self) -> None:
        pass

    def get_entities_from_falcon2(self,query,k=2):
        response = post(f"https://labs.tib.eu/falcon/falcon2/api",json={"text": query},params=(("mode","long"),("k",k)))
        entities = json.loads(response.text)["entities_wikidata"]
        return [{"id":entity[1],"label":entity[0]} for entity in entities]

    def triples_from_query(self, entity_id, entity_label):
        '''
        Formulates a sparql query, returning a simplified (networkx) kg containing the k-hop neighbourhoods of the mentioned entity
        Arguments:
            entity: tuples(string,string), tuple ,
        Returns:
            a list of triples
        '''
        fetcher = SPARQLWrapper2("https://query.wikidata.org/sparql")
        fetcher.setQuery(f'''
        SELECT DISTINCT ?item ?itemLabel ?relation ?relationLabel ?propLabel{{
            ?item ?relation {entity_id} .
            SERVICE wikibase:label {{bd:serviceParam wikibase:language "en" }}
            ?prop wikibase:directClaim ?relation .
        }}
        ''')
        triples=[]
        for result in fetcher.query().bindings:
            triples.append(({"id":result["item"],"label":result['itemLabel']},{"id":result["relation"],"label":result["propLabel"]},{"id":entity_id,"label":entity_label}))
        return triples

        # request = r'''
        # SELECT ?item ?dir1{
        #     ?item ?dir1 wd:Q6537379 .
        #     SERVICE wikibase:label {bd:serviceParam wikibase:language "en" }
        #     }
        # '''
        # self.triples = get('...',f'content={request}')
        # graph = ...
        # return graph

if __name__ == '__main__':
    de = DataExtractor(10)
    # de.triples_from_query("wd:Q6537379")
    entities = de.get_entities_from_falcon2("hello Bob Marley")
    print(entities)
    # triples = de.triples_from_query(entities[0][1],entities[0][0])
    # print(triples)
