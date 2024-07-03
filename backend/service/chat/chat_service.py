from injector import inject, singleton
from settings.settings import Settings
from component.llm.llm_component import LLMComponent
from vectorstores.chroma import Chroma
from db.mongodb import MyMongodb
from db.es_client import ElasticsearchClient
from pathlib import Path
from typing import Any, Iterable, List, Dict, Optional, Tuple
from pprint import pprint
from tools import content_title_process
from langchain.evaluation import load_evaluator
import json
import os
from langchain_community.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.evaluation.embedding_distance import EmbeddingDistanceEvalChain
from langchain.chains import RetrievalQA
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from sentence_transformers import CrossEncoder
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)


@singleton
class ChatService:
    @inject
    def __init__(
        self, 
        llm_component: LLMComponent, 
        db: MyMongodb,
        es_client: ElasticsearchClient,
        settings: Settings,
    ):
        self.llm = llm_component.llm
        self.tokenizer = self.llm.tokenizer
        self.vectorstore = Chroma(self.tokenizer, settings.chroma_collection)
        self.db = db
        self.es_client = es_client
        self.settings = settings
    
    def chat(self, message: str):
        responds = self.llm.chat(
            message, 
        )
        return responds
    
    def get_prompt_by_docs(self, question: str, docs: List[str]):
        docs_str = '\n'.join(docs)
        return f'问题：\n{question}\n文档：\n{docs_str}'

    # def chat_augmented(self, query: str, file_name: str = '', n: int = 1, **kwargs):
    #     where = {
    #         '$and': [
    #             {'file_name': {'$eq': file_name}},
    #             {'model': {'$eq': self.llm.model_name}},
    #             {'words_num': {'$gte': 30}},
    #         ],
    #     }
    #     docs1 = self.vectorstore.query_collection(query, n_results=n, where=where, **kwargs)
    #     docs2 = self.similarity_search(query, file_name=file_name, n=n, **kwargs)
    #     docs3 = self.similarity_search1(query, file_name=file_name, n=n, **kwargs)
    
    def similarity_search0(self, query: str, file_name: str = '', n: int = 1, **kwargs):
        print('--------similarity_search0----------')
        where = {
            '$and': [
                {'file_name': {'$eq': file_name}},
                {'model': {'$eq': self.llm.model_name}},
                {'words_num': {'$gte': 30}},
            ],
        }
        docs = self.vectorstore.query_collection(query, n_results=n, where=where, **kwargs)
        print(docs)
        return docs['documents'][0]
    
    def similarity_search(self, query: str, file_name: str = '', n: int = 1, **kwargs):
        print('--------similarity_search----------')
        prompt = f'为给定的问题提供一个简短的示例答案，这可能可以在一些文档中找到：{query}'
        responds = self.llm.chat(prompt)
        print(responds)
        prompt = f'{query}\n示例答案：{responds}'
        where = {
            '$and': [
                {'file_name': {'$eq': file_name}},
                {'model': {'$eq': self.llm.model_name}},
                {'words_num': {'$gte': 30}},
            ],
        }
        result = self.vectorstore.query_collection(prompt, n_results=n, where=where, **kwargs)
        print(result)
        return result['documents'][0]
    
    def similarity_search1(self, query: str, file_name: str = '', n: int = 10, **kwargs):
        print('--------similarity_search1----------')
        prompt = f'问题：{query}\n提出和这个问题最多五个相关问题来帮助他们找到他们所需要的信息，针对所提供的问题。只提简短的问题，不提复合句。提出涵盖主题不同方面的各种问题。确保它们是完整的问题，并且是相关的原来的问题。每行输出一个问题。不要给问题编号。'
        responds = self.llm.chat(prompt)
        print(responds)
        prompt = f'{query}\n{responds}'
        where = {
            '$and': [
                {'file_name': {'$eq': file_name}},
                {'model': {'$eq': self.llm.model_name}}, 
                {'words_num': {'$gte': 30}},
            ],
        }
        retrieved_documents = self.vectorstore.query_collection(prompt, n_results=n, where=where, **kwargs)
        cross_encoder = CrossEncoder(self.settings.cross_encoder_path)
        pairs = [[query, doc] for doc in retrieved_documents['documents'][0]]
        print(pairs)
        scores = cross_encoder.predict(pairs)
        result = []
        for o in np.argsort(scores)[::-1]:
            result.append(retrieved_documents['documents'][0][o])
        return result
    
    def similarity_search_by_es(self, query: str, file_name: str, n: int = 10, **kwargs):
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
                            'file_name': file_name,
                        },
                    }
                },
            },
            'size': n
        }
        docs = self.es_client.search_docs('upload_files', query_body)
        return [dos['_source']['text'] for dos in docs]
    
    def query_document_by_content(self, query: str, file_name: str):
        def get_content_in_responds() -> List[str]:
            separators = ['\n', ',', '，', '、']
            for sep in separators:
                if sep in responds:
                    titles = responds.split(sep)
                    titles = [
                        content_title_process(t)
                        for t in titles if t]
                    if len(titles) >= 1:
                        return titles
            return [responds]

        print('----------query_document---------')
        # file = Path(file_path)
        fileinfo = self.db.get_fileinfo(file_name)
        max_content_level = fileinfo.get('max_content_level')
        max_page_num = fileinfo.get('max_page_num')
        responds = ''
        index = 1
        last_titles = []
        while True:
            if index > max_content_level:
                break
            contents = self.db.get_contents(file_name, level=index, last_titles=last_titles)
            select_count = 2
            contents_str = '\n'.join(contents)
            pprint('----------contents---------')
            pprint(contents)
            # TODO 考虑将标题及所有上级标题拼一起再问LLM，因为有些标题很简单，比如“小结”
            message = f"""
                请参考以下示例回答问题：
                示例问题：
                在以下的话题中，选出{select_count}个和“什么是nosql数据库？”这个话题意思最接近的话题，直接回答原话题，不要少字，不要说多余的话：\n
                关系型数据库\n
                非关系型数据库\n
                redis使用方法\n
                母猪的产后护理\n
                \n
                示例答案：
                非关系型数据库\n
                redis使用方法
                \n
                现在回答以下问题：
                在以下的话题中，选出{select_count}个和“{query}”这个话题意思最接近的话题，直接回答原话题，不要少字，不要说多余的话：\n
                {contents_str}
                \n
            """
            responds = self.llm.chat(
                message, 
            )
            pprint('----------responds---------')
            pprint(responds)
            last_titles = get_content_in_responds()
            pprint('----------last_titles---------')
            pprint(last_titles)
            if not last_titles:
                continue
            index += 1
        print('----------------------------')
        # result_titles = []
        # for title in last_titles:
        #     t = content_title_process(title)
        #     contents = self.db.content_collection.find({
        #         'title': {'$regex': f'^.*?{t}.*'},
        #     })
        #     for content in contents:
        #         pprint(content['title'])
        #         result_titles.append(content['title'])
        contents = self.db.content_collection.find({
            'file_name': file_name
        })
        contents_info = []
        for content in contents:
            contents_info.append({
                'title': content['title'], 
                'page_num': content['page_num']
            })
        page_nums = []
        for i, content in enumerate(contents_info):
            for title in last_titles:
                t = content_title_process(title)
                if t in content['title']:
                    if i < len(contents_info):
                        page_nums.append((content['page_num'], contents_info[i + 1]['page_num']))
                    else:
                        page_nums.append((content['page_num'], max_page_num))
        pprint(page_nums)
        if not page_nums:
            return 
        where = {
            '$and': [
                {'file_name': {'$eq': file_name}},
                {'model': {'$eq': self.llm.model_name}},
                {
                    '$or': [
                        {
                            '$and': [
                                {'page_num': {'$gte': page_num[0]}}, {'page_num': {'$lte': page_num[1]}}
                            ]
                        } 
                    for page_num in page_nums]
                } 
                if len(page_nums) > 1 else 
                {
                    '$and': [
                        {'page_num': {'$gte': page_nums[0][0]}}, {'page_num': {'$lte': page_nums[0][1]}}
                    ]
                }, 
                {'words_num': {'$gte': 30}},
            ],
        }
        documents = self.vectorstore.query_collection(query, n_results=5, where=where)
        pprint(documents)

        documents_str = '\n'.join(['\n'.join([d for d in document]) for document in documents['documents']])
        query_process = f"""
            根据以下文档回答问题，<S>代表文档开始，<E>代表文档结束：
            <S>
            {documents_str}
            <E>
            现在回答问题：
            {query}
        """
        responds = self.llm.chat(
            query_process, 
        )
        return responds
    
    def evaluation_doc_by_openai(self, query: str, file_name: str, n: int = 1, **kwargs):
        print('--------evaluation_doc_by_openai----------')
        openai = ChatOpenAI(model='gpt-3.5-turbo')
        prompt = ChatPromptTemplate.from_messages([('system', '你是一个乐于助人的助手。'), ('human', '{input}')])
        chain = prompt | openai | StrOutputParser()
        openai_result = chain.invoke({'input': query})
        eval_chain = QAEvalChain.from_llm(openai)

        l = int(n / 2) 
        print(l)
        respond = self.llm.chat(query)
        docs1 = self.similarity_search0(query, file_name=file_name, n=n, **kwargs)
        respond1 = self.llm.chat(self.get_prompt_by_docs(query, docs1[:l]))
        docs2 = self.similarity_search(query, file_name=file_name, n=n, **kwargs)
        respond2 = self.llm.chat(self.get_prompt_by_docs(query, docs2[:l]))
        docs3 = self.similarity_search1(query, file_name=file_name, n=n, **kwargs)
        respond3 = self.llm.chat(self.get_prompt_by_docs(query, docs3[:l]))
        responds = [respond, respond1, respond2, respond3]

        i = 1
        result = []
        examples = []
        predictions = []
        for res in responds:
            examples.append({
                'query': query,
                'answer': openai_result,
            })
            predictions.append({
                'result': res
            })
            i += 1
        graded_outputs = eval_chain.evaluate(examples, predictions)
        for i, eg in enumerate(examples):
            result.append({'score': graded_outputs[i]['text'], 'query': eg['query'], 'answer': eg['answer'], 'respond': predictions[i]['result']})
        return result

    def evaluation_doc_by_distance(self, query: str, answer: str, file_name: str, n: int = 1, **kwargs):
        embeddings = FakeEmbeddings(model=self.settings.llm_model_path, size=self.settings.llm_size)
        evaluator = EmbeddingDistanceEvalChain(embeddings=embeddings)

        l = int(n / 2) 
        respond = self.llm.chat(query)
        docs1 = self.similarity_search0(query, file_name=file_name, n=n, **kwargs)
        query1 = self.get_prompt_by_docs(query, docs1[:l])
        respond1 = self.llm.chat(query1)
        docs2 = self.similarity_search(query, file_name=file_name, n=n, **kwargs)
        query2 = self.get_prompt_by_docs(query, docs2[:l])
        respond2 = self.llm.chat(query2)
        docs3 = self.similarity_search1(query, file_name=file_name, n=n, **kwargs)
        query3 = self.get_prompt_by_docs(query, docs3[:l])
        respond3 = self.llm.chat(query3)
        querys = [query, query1, query2, query3]
        responds = [respond, respond1, respond2, respond3]

        result = []
        for i, res in enumerate(responds):
            score = evaluator.evaluate_strings(prediction=res, reference=answer)
            result.append({'score': score.get('score'), 'query': querys[i], 'answer': answer, 'respond': res})
        return result

    def evaluation_test_dateset_by_distance(self, test_file: str, n: int) -> List[Dict]:
        embeddings = FakeEmbeddings(model=self.settings.llm_model_path, size=self.settings.llm_size)
        # evaluator = load_evaluator('embedding_distance')
        evaluator = EmbeddingDistanceEvalChain(embeddings=embeddings)
        i = 1
        result = []
        with open(test_file, 'r') as f:
            for l in f.readlines():
                if (i > n) and n:
                    break
                line = json.loads(l)
                user_value = line['messages'][1]['content']
                assistant_value = line['messages'][2]['content']
                respond = self.llm.chat(user_value)
                score = evaluator.evaluate_strings(prediction=respond, reference=assistant_value)
                result.append({'score': score.get('score'), 'query': user_value, 'answer': assistant_value, 'respond': respond})
                i += 1
        return result
    
    def evaluation_test_dateset_by_openai(self, test_file: str, n: int) -> List[Dict]:
        openai = ChatOpenAI(model='gpt-3.5-turbo')
        eval_chain = QAEvalChain.from_llm(openai)
        i = 1
        result = []
        examples = []
        predictions = []
        with open(test_file, 'r') as f:
            for l in f.readlines():
                if (i > n) and n:
                    break
                line = json.loads(l)
                user_value = line['messages'][1]['content']
                assistant_value = line['messages'][2]['content']
                respond = self.llm.chat(user_value)
                examples.append({
                    'query': user_value,
                    'answer': assistant_value,
                })
                predictions.append({
                    'result': respond
                })
                i += 1
        graded_outputs = eval_chain.evaluate(examples, predictions)
        for i, eg in enumerate(examples):
            result.append({'score': graded_outputs[i]['text'], 'query': eg['query'], 'answer': eg['answer'], 'respond': predictions[i]['result']})
        return result

