from torch.nn import Module
from models.luke.entity_disambiguation import LukeForEntityDisambiguation
### model from: "Global Entity Disambiguation with Pretrained Contextualized Embeddings of Words and Entities" - Yamada 2020

class CandidateGenerator(Module):
    def __init__(self,candidate_size):
        super().__init__()
        self.candidate_size=candidate_size

    def _similarity(self, word1, word2):
        pass

    def _topk(self,scores):
        pass

    def forward(self, query, entities):
        '''
        Arguments:
            Query: list[mentions]
            Entities: dict[alias -> entity_id]
        Returns:
            tensor(candidate_size,)
        '''
        scores={} #dict[entity_id->match_score]
        for mention in query:
            for alias,entity_id in entities.items():
                score = self._similarity(mention,alias)
                if entity_id in scores:
                    scores[entity_id]=max(scores[entity_id],score)
                else:
                    scores[entity_id]=score
        entity_ids = self._topk(scores)
        return entity_ids

class EntityDisambiguator(Module):
    def __init__(self,luke_checkpoint):
        super().__init__()
        self.model = LukeForEntityDisambiguation()
        self.model.load_state_dict(luke_checkpoint)

    def forward(self,question, entities):
        '''
        Given a list 
        Arguments:
            question: [ca]
            entities: [candidate entities] 
        Returns:
            disambiguj
        '''
