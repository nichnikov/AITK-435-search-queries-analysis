"""
классификатор KNeighborsClassifier в /home/an/Data/Yandex.Disk/dev/03-jira-tasks/aitk115-support-questions
"""
import re
from src.data_types import (Parameters, 
                            SearchResult)
from src.storage import ElasticClient
from src.texts_processing import TextsTokenizer
from src.config import (logger, 
                        empty_result)
from sentence_transformers import util
from collections import namedtuple
from src.utils import jaccard_similarity

# https://stackoverflow.com/questions/492519/timeout-on-a-function-call

def texts_entrance(incoming_text: str, receiving_text: str):
    print(incoming_text, receiving_text, re.findall(incoming_text, receiving_text))
    if re.findall(incoming_text, receiving_text):
        return True
    else:
        return False

def search_result_rep(search_result: []):
    return [{**d["_source"],
             **{"id": d["_id"]},
             **{"score": d["_score"]}} for d in search_result]


class FastAnswerClassifier:
    """Объект для оперирования MatricesList и TextsStorage"""

    def __init__(self, tokenizer: TextsTokenizer, parameters: Parameters, model):
        self.es = ElasticClient()
        self.tkz = tokenizer
        self.prm = parameters
        self.model = model
        self.topic_check = True
    
    async def get_answer(self, templateId, pubid, score, topic, jaccard, entrancy, best_etalon):
        # answer_query = {"bool": {"must": [{"match_phrase": {"templateId": templateId}}, {"match_phrase": {"pubId": pubid}},]}}
        answer_query = {"match_phrase": {"templateId": templateId}}
        resp = await self.es.search_by_query(self.prm.answers_index, answer_query)
        if resp["hits"]["hits"]:
            search_result = search_result_rep(resp["hits"]["hits"])
            return SearchResult(templateId=search_result[0]["templateId"], templateText=search_result[0]["templateText"], topic=topic, 
                                score=score, jaccard=jaccard, entrance=entrancy, best_etalon=best_etalon).dict()
        else:
            logger.info("not found answer with templateId {} and pub_id {}".format(str(templateId), str(pubid)))
            return empty_result

    async def searching(self, text: str, pubid: int, score: float, candidates: int):
        
        """searching etalon by  incoming text"""
        try:
            tokens = self.tkz([text])
            if tokens[0]:
                tokens_str = " ".join(tokens[0])
                query = {"match": {"LemCluster": tokens_str}}
                search_result = await self.es.search_by_query(self.prm.clusters_index, query)
                if search_result["hits"]["hits"]:
                    etalons_search_result = search_result_rep(search_result["hits"]["hits"])
                    if etalons_search_result:
                        results_tuples = [(d["ID"], d["Cluster"], d["LemCluster"], d["Topic"]) for d in etalons_search_result[:candidates]]
                        # print(results_tuples[:15])
                        # print(len(results_tuples))
                        text_emb = self.model.encode(tokens_str)
                        ids, ets, lm_ets, topics = zip(*results_tuples)
                        candidate_embs = self.model.encode(lm_ets)
                        scores = util.cos_sim(text_emb, candidate_embs)
                        scores_list = [score.item() for score in scores[0]]
                        lm_topics = [" ".join(lm_tp) for lm_tp in self.tkz(topics)]
                        if self.topic_check:
                            """дополнительное условие на Топики:"""
                            '''
                            for lm_tp in lm_topics:
                                res = texts_entrance(lm_tp, tokens_str)
                                print (lm_tp, tokens_str, res)'''
                            etalons_candidates = [(i, et, lm_et, lm_tp, sc, jaccard_similarity(lm_tp, tokens_str),  texts_entrance(lm_tp, tokens_str)) for i, et, lm_et, lm_tp, sc in
                                                zip(ids, ets, lm_ets, lm_topics, scores_list)  if sc >= score]
                        else:
                            """отсуствие условия на топики:"""
                            etalons_candidates = [(i, et, lm_et, lm_tp, sc, 0.0, False) for i, et, lm_et, lm_tp, sc in zip(ids, ets, lm_ets, lm_topics, scores_list)
                                                  if sc >= score]
                        the_best_result = sorted(etalons_candidates, key=lambda x: x[4], reverse=True)[0]

                        if the_best_result:
                            answer = await self.get_answer(the_best_result[0], pubid, the_best_result[4], the_best_result[3], 
                                                           the_best_result[5], the_best_result[6], the_best_result[1])
                            logger.info("search completed successfully with result: {}".format(str(the_best_result)))
                            return answer
                        else:
                            logger.info(
                                "not found answer with templateId {} and pub_id {}".format(str(the_best_result[0]),
                                                                                        str(pubid)))
                            return empty_result
                    else:
                        logger.info("elasticsearch doesn't find any etalons for input text {}".format(str(text)))
                        return empty_result
                else:
                    logger.info("elasticsearch doesn't find any etalons for input text {}".format(str(text)))
                    return empty_result
            else:
                logger.info("tokenizer returned empty value for input text {}".format(str(text)))
                return empty_result
        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            return empty_result
        

from src.config import logger
from collections import namedtuple


# https://stackoverflow.com/questions/492519/timeout-on-a-function-call

class TFIDFClassifier:
    """Объект для оперирования MatricesList и TextsStorage"""

    def __init__(self,
                 tokenizer,
                 parameters,
                 gensim_dict,
                 tfidf_model,
                 gensim_index,
                 answers):

        self.tkz = tokenizer
        self.prm = parameters
        self.tfidf = tfidf_model
        self.dct = gensim_dict
        self.index = gensim_index
        self.answers = answers

    async def searching(self, text: str):
        """searching etalon by  incoming text"""
        try:
            tokens = self.tkz([text])
            if tokens[0]:
                in_corpus = self.dct.doc2bow(tokens[0])
                in_vector = self.tfidf[in_corpus]
                sims = self.index[in_vector]
                # print([(num, scr) for num, scr in enumerate(list(sims), start=1)])
                tfidf_tuples = [(num, scr) for num, scr in enumerate(list(sims), start=1) if scr >= self.prm.score]
                if tfidf_tuples:
                    tfidf_best = sorted(tfidf_tuples, key=lambda x: x[1], reverse=True)[0]
                    best_id = tfidf_best[0]
                    templateText = self.answers[best_id]
                    search_result = {"templateId": best_id, "templateText": templateText}
                    logger.info("search completed successfully with result: {}".format(str(search_result)))
                    return search_result
                else:
                    return {"templateId": 0, "templateText": ""}
            else:
                return {"templateId": 0, "templateText": ""}
        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            return {"templateId": 0, "templateText": ""}