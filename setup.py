import urllib.request

if __name__ == '__main__':
    urllib.request.urlretrieve("https://storage.googleapis.com/gresearch/kelm-corpus/updated-2021/kelm_generated_corpus.jsonl","Datasets/KELMCorpus/kelm_generated_corpus.jsonl")