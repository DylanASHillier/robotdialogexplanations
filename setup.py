import urllib.request
import gzip
import nltk

if __name__ == '__main__':
    # urllib.request.urlretrieve("https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/kelm_generated_corpus.jsonl","datasets/KELMCorpus/kelm_generated_corpus.jsonl")
    # urllib.request.urlretrieve("https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.gz","datasets/KGs/wikidata_dump.gz")
    # urllib.request.urlretrieve("https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz","datasets/KGs/conceptnetassertions.csv.gz")
    # nltk.download("punkt")
    # nltk.download("averaged_perceptron_tagger")
    # nltk.download("universal_tagset")
    # urllib.request.urlretrieve("https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip", "datasets/scene_graphs/questions1.2.zip")
    urllib.request.urlretrieve("https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip","datasets/scene_graphs/sceneGraphs.zip")