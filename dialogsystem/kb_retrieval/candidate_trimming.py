from torch.nn import Module
import Levenshtein
import nltk
from sentence_transformers import SentenceTransformer
from torch import tensor, dot


class CandidateGenerator():
    '''
    Class for finding words that are similar to entities in text, given a list of entities.
    Intended to be used with smaller KGs that have a reasonable amount of entities to do this over.
    '''
    def __init__(self,candidate_size,candidate_threshold=0.8):
        '''
        candidate_size: int, the maximum number of candidates output
        candidate_threshold: float?, minimum score on similarity metric to be used
        '''
        super().__init__()
        self.candidate_size=candidate_size
        self.candidate_threshold=candidate_threshold
        self.scores={}  #dict[entity_id->match_score]
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def _similarity(self, word1, word2):
        '''
        Given strings word1, word2 returns a metric of distance between them
        '''
        return Levenshtein.ratio(word1.lower(),word2.lower())

    def _topk(self):
        '''
        Uses the self.scores dictionary and returns the self.candidate_size - top scoring entity_id's
        '''
        return sorted(self.scores.copy(), key=self.scores.get, reverse=True)[:self.candidate_size]

    def transformer_trim(self, query, entities):

        sentences = [query]+[alias for alias in entities.keys()]

        embeddings = tensor(self.model.encode(sentences))
        for i in range(1,len(sentences)):
            score = dot(embeddings[0],embeddings[i])
            if self.candidate_threshold is None or score > self.candidate_threshold:
                if entities[sentences[i]] in self.scores:
                    self.scores[entities[sentences[i]]]=max(self.scores[entities[sentences[i]]],score)
                else:
                    self.scores[entities[sentences[i]]]=score
        entity_ids = self._topk()
        self.scores = {} ## reset scores
        return entity_ids

    def trim(self, query, entities):
        '''
        Arguments:
            Query: string
            Entities: dict[alias -> entity_id]
        Returns:
            list[entity_ids]
        '''
        tokens = nltk.word_tokenize(query)
        tagged = nltk.pos_tag(tokens,tagset='universal')
        mentions = []
        for word,part in tagged:
            if part in ["ADJ","ADP","NOUN","NUM","VERB","X","PRON"]:
                mentions.append(word)
        for mention in mentions:
            for alias,entity_id in entities.items():
                for word in alias.split():
                    score = self._similarity(mention,word)
                    if self.candidate_threshold is None or score > self.candidate_threshold:
                        if entity_id in self.scores:
                            self.scores[entity_id]=max(self.scores[entity_id],score)
                        else:
                            self.scores[entity_id]=score
        entity_ids = self._topk()
        self.scores = {} ## reset scores
        return entity_ids