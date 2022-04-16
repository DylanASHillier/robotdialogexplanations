from rdflib import Graph, Literal
from networkx import DiGraph as nxDiGraph
from networkx import Graph as nxUdGraph
import rdflib
from requests import get, post
from SPARQLWrapper import SPARQLWrapper2
import json
from urllib.error import HTTPError
import time
import tqdm

def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    ... Copied from https://github.com/INK-USC/KagNet
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s

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

    def formulate_wikidata_sparql_query(self, entities):
        header = '''
        SELECT DISTINCT ?item ?itemLabel ?relation ?relationLabel ?propLabel ?value ?valueLabel WHERE
        {
            ?item ?relation ?value .
            SERVICE wikibase:label {bd:serviceParam wikibase:language "en". }
            ?prop wikibase:directClaim ?relation .
            VALUES (?value) {('''
        values = ")\n\t\t(".join([entity["id"] for entity in entities])
        tail = ''')
            }
        }
        LIMIT 10000'''
        return header+values+tail

    # def formulate_local_sparql_query(self, entity1,entity2, k):
    #     f'''
    #     CONSTRUCT {{ ?subj ?pred ?obj }}
    #     WHERE {{
    #         {entity1} ?pred0 ?subj .
    #         ?subj ?pred ?obj
    #     }}
    #     '''

    def triples_from_wikidata_query(self, entities):
        '''
        Formulates a sparql query, returning a simplified (networkx) kg containing the k-hop neighbourhoods of the mentioned entity
        Arguments:
            entities: list[dict{id:string,entity_label:string}]
        Returns:
            a list of triples
        '''
        fetcher = SPARQLWrapper2("https://query.wikidata.org/sparql")
        fetcher.setQuery(
            self.formulate_wikidata_sparql_query(entities)
        )
        triples=[]
        while len(triples)==0:
            try:
                for result in fetcher.query().bindings:
                    triples.append(({"id":result["item"].value,"label":result['itemLabel'].value},{"id":result["relation"].value,"label":result["propLabel"].value},{"id":result["value"].value,"label":result["valueLabel"].value}))
            except HTTPError as err:
                print(f"http error, waiting then trying again {err}")
                time.sleep(20)

        return triples

    def triples_etc_from_conceptnet_csv(self, csv, target='networkx', directed=True):
        '''
        Copied from https://github.com/INK-USC/KagNet
        returns relations, entities, triples for conceptnet

        arguments:
            csv: source csv,
            filepath: where to store
            target: string from {networkx,rdflib}
        '''
        # triples = []
        relations = set()
        entities = set()
        if target == 'rdflib':
            g = Graph()
        else:
            if directed:
                g = nxDiGraph()
            else:
                g = nxUdGraph()
        with open(csv, 'r', encoding="utf8") as f:
            for line in tqdm.tqdm(f):
                ls = line.split('\t')
                if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en/'):
                    """
                    Some preprocessing:
                        - Remove part-of-speech encoding.
                        - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                        - Lowercase for uniformity.
                    """
                    rel = ls[1].split("/")[-1].lower()
                    head = del_pos(ls[2]).split("/")[-1].lower()
                    tail = del_pos(ls[3]).split("/")[-1].lower()

                    if not head.replace("_", "").replace("-", "").isalpha():
                        continue
                    if not tail.replace("_", "").replace("-", "").isalpha():
                        continue
                    if rel.startswith("*"):
                        rel = rel[1:]
                        tmp = head
                        head = tail
                        tail = tmp
                    relations.add(rel)
                    entities.add(head)
                    entities.add(tail)
                    if target == 'rdflib':
                        g.add((Literal(head),Literal(rel),Literal(tail)))
                    else:
                        g.add_edge(head,tail,label=rel)
                    # triples.append(({"id":head,"label":head},{"id":rel,"label":rel},{"id":tail,"label":tail}))
        print(len(relations),len(entities))
        return relations, entities, g

    # def triples_from_rdflib(self, entities):
    #     g = rdflib.Graph()
    #     for entity in entities:
    #         g.parse(entity["id"][1:-1]+'.n3')
    #     qres = g.query(self.formulate_sparql_query(entities))
    #     triples =[]
    #     for result in qres.bindings:
    #         triples.append(({"id":result["item"],"label":result['itemLabel']},{"id":result["relation"],"label":result["propLabel"]},{"id":result["value"],"label":result["valueLabel"]}))
    #     return triples


if __name__ == '__main__':
    de = DataExtractor()
    # de.triples_from_query("wd:Q6537379")
    # entities = de.get_entities_from_falcon2("hello Bob Marley")
    entities = de.triples_from_csv("datasets/KGs/conceptnetassertions.csv")
    print(entities)
    # triples = de.triples_from_query(entities[0][1],entities[0][0])
    # print(triples)
