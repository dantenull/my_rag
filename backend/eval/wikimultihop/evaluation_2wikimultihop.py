from .evaluate_2wikimultihop import eval
from component.llm.llm_component import LLMComponent
from component.embeddings.embeddings_component import EmbeddingsComponent
from settings.settings import Settings
from db.mongodb import MyMongodb
from db.es_client import ElasticsearchClient
from celery_app import process_async_insert_to_es, insert_to_milvus, insert_chunk_to_mongodb
from vectorstores.milvus import Milvus
from query_optimizer import *
from tools import reciprocal_rank_fusion

import json
from datetime import datetime
from celery import group, chain
from typing import List, Tuple, Dict
from injector import singleton
import uuid
from pathlib import Path

# def reciprocal_rank_fusion(search_results_dict: List[List], k: int = 60) -> List[Tuple]:
#     fused_scores = {}
#     for docs in search_results_dict:
#         for rank, doc in enumerate(docs):
#             doc_id, text, doc_name = doc
#             if doc_id not in fused_scores:
#                 fused_scores[doc_id] = (0, text, doc_name)
#             # previous_score = fused_scores[doc]
#             fused_scores[doc_id][0] += 1 / (rank + k)
#             # print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

#     reranked_results = [(doc_id, score[0], score[1], score[2]) for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)]
#     print("Final reranked results:", reranked_results)
#     return reranked_results


@singleton
class Evaluation2wikimultihop:
    def __init__(
        self, 
        llm_component: LLMComponent, 
        embeddings_component: EmbeddingsComponent, 
        db: MyMongodb,
        es_client: ElasticsearchClient,
        settings: Settings, 
        # file_path: str,
        # eval_num: int=100,
    ) -> None:
        self.settings = settings
        self.llm = llm_component.llm
        self.embeddings = embeddings_component.embeddings
        self.vectorstore = Milvus(settings.milvus.uri, settings.milvus.port, settings.milvus.database)
        self.db = db
        self.es_client = es_client
        self.embeddings_model = settings.embeddings.model_name
        # self.eval_num = eval_num
        # self.file_path = file_path
        # self._load_dataset(file_path / 'gold.json', eval_num)
    
    def init(self, file_path: str, eval_num: int=100):
        self.file_path = Path(file_path)
        self._load_dataset(self.file_path / 'gold.json', eval_num)
    
    def _load_dataset(self, file_path: str, eval_num: int=100):
        with open(file_path, 'r', encoding='utf8') as f:
            content = json.loads(f.read())
        if eval_num:
            content = content[:eval_num]
        self.origin_data = content
        self.db.insert_data_by_field('2wikimultihop_data', self.origin_data, '_id')
    
    def _process_data(self, data: Dict):
        file_id = data['_id']
        question = data['question']
        contexts = data['context']
        type = data['type']

        data = self.db.get_one_data('files', {'file_id': file_id})
        if data:
            return

        info = {
            'file_id': file_id,
            # 'name': file_id,
            'upload_date': datetime.now(),
            'upload_state': 'waiting',
            'upload_state_no_sql': 'done',
            'upload_state_elasticsearch': 'waiting',
            'upload_state_vectorstore': 'waiting',
            # 'title': file_id,
            'embedding_model': self.embeddings_model,
            'type': type,
            'question': question,
        }
        self.db.insert_fileinfo(info)

        documents = []
        chunks = []
        for context in contexts:
            doc_name = context[0]
            texts = context[1]
            for i, text in enumerate(texts):
                doc_id = str(uuid.uuid4())
                documents.append({
                    'file_id': file_id,
                    'doc_id': doc_id,
                    'text': text,
                    'embedding_model': self.embeddings_model,
                    'metadata': {
                        'index': i,
                        'doc_name': doc_name,
                    }
                })
                chunks.append({
                    'file_id': file_id,
                    'file_name': file_id,
                    'id': doc_id,
                    'chunk': text,
                    'doc_name': doc_name,
                    'document': '\n'.join(texts),
                    'index': i,
                    'metadata': {
                        'doc_name': doc_name,
                        'embeddings_model': self.embeddings_model,
                        'file_name': file_id,
                        'seq': i
                    }
                })
        task_es = process_async_insert_to_es.apply_async(kwargs={
            'file_id': file_id, 
            'index_name': 'upload_files', 
            'data': documents,
            'task_start': True,
        })
        # for context in contexts:
        #     doc_name = context[0]
        #     documents = context[1]
        #     for document in documents:
        #         chunks.append({
        #             'chunk': document,
        #             'document': '\n'.join(documents),
        #             'embeddings_model': self.embeddings_model,
        #             'file_id': file_id,
        #         })
        task_vectorstore = chain(
            group([insert_chunk_to_mongodb.si(chunk_info=chunk, file_id=file_id) for chunk in chunks]), 
            insert_to_milvus.si(file_id = file_id, task_end = True)
        ).apply_async()

    def process_datas(self):
        for data in self.origin_data:
            self._process_data(data)

    def _similarity_search_by_vectordb(self, query: str, file_id: str = '', n: int = 10, **kwargs):
        # print('--------similarity_search_by_vectordb----------')
        # query = QueryOptimiserQuery2doc().process_query(query)
        data = self.embeddings.encode(query)
        result = self.vectorstore.search_data(self.settings.embeddings.dim, data=data, limit=n)
        collection_name = f'vector_{self.settings.embeddings.dim}'
        chunks = self.vectorstore.client.get(collection_name, [r['id'] for r in result], output_fields=['chunk', 'doc_name', 'seq'])
        # print([(chunk.get('id'), chunk.get('doc_name'), chunk.get('seq')) for chunk in chunks])
        return [(chunk['id'], chunk['chunk'], [chunk['doc_name'], chunk['seq']]) for chunk in chunks]
    
    async def _similarity_search_by_es(self, query: str, file_id: str, n: int = 10, **kwargs):
        query_body = {
            'query': {
                'bool': {
                    'must': {
                        'match': {
                            'text': {
                                'query': query,
                                'fuzziness': 'AUTO',
                            }
                        },
                    },
                    'filter': {
                        'match': {
                            'file_id': file_id,
                        },
                    }
                },
            },
            'size': n
        }
        docs = await self.es_client.async_search_docs('upload_files', query_body)
        # print(docs[:n])
        return [(doc['_source']['doc_id'], doc['_source']['text'], [doc['_source']['doc_name'], doc['_source']['index']]) for doc in docs[:n]]
    
    def _get_rrf_result(self, result: List[List], n: int) -> List[Tuple]:
        reranked_results = reciprocal_rank_fusion(result)
        if len(reranked_results) > n:
            reranked_results = reranked_results[:n]
        texts = [(r[2], r[3]) for r in reranked_results]
        return texts
    
    async def _mix_search(self, texts: List[str], query: str) -> str:
        # vectordb_result = self._similarity_search_by_vectordb(query, file_id, n, **kwargs)
        # es_result = await self._similarity_search_by_es(query, file_id, n, **kwargs)
        # texts = self._get_rrf_result([vectordb_result, es_result], n)
        # prompt = 'According to the provided document, answer the following questions:\n' + '\n'.join(texts)
        prompt = f"Based on the given document, answer this question:{query}\nPlease follow the following guidelines when answering:\nYou don't need to explain the reason. Just give a brief answer."
        query = 'The given document is as follows:' + '\n'.join(texts)
        result = self.llm.chat(query=query, prompt=prompt)
        return result
    
    async def _get_answer(self, data: Dict):
        evidence = []
        file_id = data['_id']
        question = data['question']
        # supporting_facts = data['supporting_facts']
        # evidences = data['evidences']
        n = 5
        vectordb_result = self._similarity_search_by_vectordb(question, file_id, n)
        es_result = await self._similarity_search_by_es(question, file_id, n)
        texts = self._get_rrf_result([vectordb_result, es_result], n)
        search_result = [texts, vectordb_result, es_result]
        sp = [text[1] for text in texts]
        answer = await self._mix_search([text[0] for text in texts], question)
        return answer, sp, evidence, search_result

    async def eval_dataset(self):
        result = {
            'answer': {},
            'sp': {},
            'evidence': {},
        }
        a = len(self.origin_data)
        i = 0
        eval_id = str(uuid.uuid4())
        result1 = []
        for data in self.origin_data:
            try:
                file_id = data['_id']
                answer, sp, evidence, search_result = await self._get_answer(data)
                result['answer'][file_id] = answer
                result['sp'][file_id] = sp
                result['evidence'][file_id] = evidence
                result1.append({
                    'eval_id': eval_id,
                    'file_id': file_id,
                    'answer': answer,
                    'sp': sp,
                    'evidence': evidence,
                    'search_result': search_result
                })
            except Exception as e:
                print(str(e))
                continue
            i += 1
            print(f'{i} / {a}')
            # if i > 1000:
            #     break
        
        self.db.insert_data_by_field('eval_results_2wikimultihop', result1, 'file_id')
            
        # prediction_file = './pred.json'
        # with open(prediction_file, 'w') as f:
        #     f.write(json.dumps(result))
        # metrics = eval(prediction_file, self.file_path / 'gold.json', self.file_path / 'id_aliases.json')

        metrics = eval(result, self.origin_data, self.file_path / 'id_aliases.json')

        self.db.insert_data('eval_result', {
            'dataset': '2wikimultihop',
            'result': metrics,
            'create_time': datetime.now(),
            'eval_id': eval_id,
            'eval_num': i,
        })
        return metrics
