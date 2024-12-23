from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from injector import singleton, inject
import json
from query_optimizer import *
from settings.settings import unsafe_typed_settings as rag_settings
from tools import get_embedding_model
from vectorstores.chroma import Chroma
from reader.parser.sentence import SentenceSplitter
import uuid
from typing import List, Tuple, Dict
from datasets import Dataset, Features, Sequence, Value
from rerank.rerank import Rerank
from ..evaluation import Evaluation
from component.llm.llm_component import LLMComponent
from component.embeddings.embeddings_component import EmbeddingsComponent
from settings.settings import Settings
from db.mongodb import MyMongodb
from db.es_client import ElasticsearchClient
from celery_app import process_async_insert_to_es, insert_to_milvus, insert_chunk_to_mongodb
from vectorstores.milvus import Milvus
# from query_optimizer import *
from tools import reciprocal_rank_fusion


@singleton
class EvaluationQuality(Evaluation):
    # @inject
    def __init__(
        self, 
        llm_component: LLMComponent, 
        embeddings_component: EmbeddingsComponent, 
        db: MyMongodb,
        es_client: ElasticsearchClient,
        settings: Settings, 
    ) -> None:
        super().__init__(llm_component, embeddings_component, db, es_client, settings)
        # self.llm = LLMComponent(rag_settings).llm
        # self.tokenizer = self.llm.tokenizer
        # # TODO 改
        # self.embedding_model = get_embedding_model(
        #     rag_settings.embeddings.model, rag_settings.embeddings.model_name, self.llm.tokenizer)
        # self.vectorstore = Chroma(self.tokenizer, 'eval_by_quality', self.embedding_model)
        # self.load_dataset_quality(eval_num)
        # self.cross_encoder_path = rag_settings.rerank.cross_encoder_path
    
    def init(self, eval_num: int=100):
        self._load_dataset(eval_num)

    def _load_dataset(self, eval_num: int=100):
        '''
        加载QuALITY数据集。
        参数:
        eval_num (int): 加载数据质量评估的数量，默认为 100
        '''
        # print('load_dataset_quality')
        file_path = '.\\backend\\eval\\QuALITY.v1.0.1.htmlstripped.train'
        data = []
        with open(file_path, 'r', encoding='utf8') as f:
            i = 0
            for line in f:
                if (i >= eval_num) and (eval_num > 0):
                    break
                line_data = json.loads(line)
                data.append(line_data)
                i += 1
        self.origin_data = data
        # print(data[:1])

    def _split_text(self, text: str, title: str) -> Tuple[List]:
        # print('_split_text')
        texts = SentenceSplitter().split_text(text)
        ids = []
        documents = []
        metadatas = []
        print(title)
        result = self.vectorstore.get({"title": {'$eq': title}})
        # print(result)
        if result['ids']:
            return ids, documents, metadatas
        for text in texts:
            doc_id = str(uuid.uuid4())
            metadata = {
                'title': title,
                'embedding_model': self.embedding_model,
            }
            ids.append(doc_id)
            documents.append(text)
            metadatas.append(metadata)
        # print(len(documents), len(metadatas), len(ids))
        # 之所以在这里就存向量数据库，是因为如果在之后一次性存所有的话，显存可能不够。
        # TODO 判断是否在本地使用显卡进行embedding。
        self.vectorstore.add_texts(documents, metadatas, ids)  
        return ids, documents, metadatas
    
    def _process_data(self):
        '''
        将QuALITY数据集质量评估数据转换为ragas的评估数据格式。
        并将数据加入进向量数据库。
        '''
        data = {
            'question': [],
            'ground_truth': [],
            'contexts': [],
            'answer': [],
            'title': [],
        }
        # ids = []
        # documents = []
        # metadatas = []
        for d in self.origin_data:
            ids1, documents1, metadatas1 = self._split_text(d['article'], d['title'])
            # ids.extend(ids1)
            # documents.extend(documents1)
            # metadatas.extend(metadatas1)
            for question in d['questions']:
                data['question'].append(question['question'])
                data['ground_truth'].append(question['options'][question['gold_label'] - 1])
                data['title'].append(d['title'])
                # data.append({
                #     'question': question['question'],
                #     'ground_truth': question['options'][question['gold_label'] - 1],
                #     # 'contexts': [d['article']],
                #     'title': d['title'],
                # })
        self.process_data = data
        # print(len(documents), len(metadatas), len(ids))
        # if ids:
        #     self.vectorstore.add_texts(documents, metadatas, ids)
        # print(data[:3])

    def _get_answer(self, title: str, question: str, query_optimizer: QueryOptimizer):
        '''
        通过QueryOptimizer对问题进行优化，然后通过向量数据库检索文档，最后通过LLM生成答案。
        '''
        # print('_get_answer')
        querys = query_optimizer.process_query(question)
        where = {'title': {'$eq': title}}
        context = ''
        contexts = []
        ds = []
        for query in querys:
            docs = self.vectorstore.query_collection(query, n_results=20, where=where)['documents'][0]
            ds.extend(docs)
        ds = Rerank(self.cross_encoder_path).rerank(question, ds)
        for d in ds[:5]:    
            contexts.append(d)
        if not contexts:
            raise ValueError('No context found')
        context = '\n'.join(contexts)
        prompt = (
            f"We have provided context information below. \n"
            f"---------------------\n"
            f"{context}"
            f"\n---------------------\n"
            f"Given this information, please answer the question: {question}"
        )
        result = self.llm.chat(prompt)
        # print(result)
        return result, contexts
    
    def _get_answer_quality(self):
        '''
        将answer加入进ragas的评估数据格式
        '''
        for i, title in enumerate(self.process_data['title']):
            anwser, contexts = self._get_answer(title, self.process_data['question'][i], QueryOptimiserMultiQuery())
            self.process_data['answer'].append(anwser) 
            self.process_data['contexts'].append(contexts) 
        # del self.process_data['title']
        # print(self.process_data[:3])
    
    def eval_dataset_quality(self):
        '''
        基于QuALITY数据集进行评估。
        '''
        self._process_data()
        self._get_answer_quality()
        # features = Features({
        #     "contexts": Sequence(Value("string")),
        #     "question": Value("string"),
        #     "ground_truth": Value("string"),
        #     "answer": Value("string"),
        # })
        result = evaluate(
            Dataset.from_dict(self.process_data),
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
        )
        return result





