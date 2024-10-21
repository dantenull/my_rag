from typing import Union, Optional
from component.llm.llm_component import LLMComponent
from settings.settings import unsafe_typed_settings as rag_settings


class QueryOptimizer:
    '''
    只针对 query 进行处理，不涉及到后续向量化以及检索文档。
    '''
    name = ''
    prompt_tmpl = ''

    def __init__(self) -> None:
        self.llm = LLMComponent(rag_settings).llm

    def process_query(self, query: str) -> Union[list[str], str]:
         return self._process_query(query)

    def _process_query(self, query: str) -> str:
        resp = self.llm.chat(self.prompt_tmpl.format(query=query))
        return resp


class QueryOptimiserHyDE(QueryOptimizer):
    '''
    首先，HyDE 针对 query 直接生成一个假设性文档或者说回答（hypo_doc）。
    然后，对这个假设性回答进行向量化处理。最后，使用向量化的假设性回答去检索相似文档。

    参考 llamaindex 的 HyDEQueryTransform 类。
    '''
    name = 'hyde'
    # prompt_tmpl = (
    #     "Please write a passage to answer the question\n"
    #     "Try to include as many key details as possible.\n"
    #     "\n"
    #     "\n"
    #     "{query}\n"
    #     "\n"
    #     "\n"
    #     'Passage:"""\n'
    # )
    prompt_tmpl = (
        "请写一篇短文来回答这个问题\n"
        "试着包括尽可能多的关键细节。\n"
        "\n"
        "\n"
        "{query}\n"
        "\n"
        "\n"
        '短文："""\n'
    )

    # def process_query(self, query: str) -> list[str]:
    #     return [self._process_query(query)]


class QueryOptimiserMultiQuery(QueryOptimizer):
    '''
    通过生成多种视角的查询来检索相关文档。

    参考 langchain 的 MultiQueryRetriever 类。
    '''
    name ='multi_query'
    # prompt_tmpl = """You are an AI language model assistant. Your task is 
    # to generate 3 different versions of the given user 
    # question to retrieve relevant documents from a vector database. 
    # By generating multiple perspectives on the user question, 
    # your goal is to help the user overcome some of the limitations 
    # of distance-based similarity search. Provide these alternative 
    # questions separated by newlines. Original question: {query}"""
    prompt_tmpl = """你是一个人工智能语言模型助手。你的任务是生成
    给定用户的3个不同版本从矢量数据库中检索相关文档的问题。
    通过对用户问题产生多种视角，您的目标是帮助用户克服一些限制基于距离的相似度搜索。
    提供这些选择用换行符分隔的问题。最初的问题： {query}"""

    def process_query(self, query: str) -> list[str]:
        resp = self._process_query(query)
        lines = resp.strip().split("\n")
        return lines


class QueryOptimiserQuery2doc(QueryOptimizer):
    '''
    在开始检索之前，先用query让模型生成一次答案，然后把query和答案合并送给模型。
    首先是字面检索，因为模型的生成多半会很长，所以在相似度计算的过程中，会稀释，所以在拼接过程中，需要对原始query复制几份再来拼接。复制5次通常是一个不错的值。
    然后是向量检索，因为向量召回的泛化能力是比较强的，因此不需要复制，直接拼接起来就好了。
    '''
    name = 'query2doc'
    pormpt_tmpl = '{query}\n\n要求：用大约100字回复以上问题。'
    # copy_count = 5

    # def process_query(self, query: str) -> list[str]:
    #     resp = self._process_query(query)
    #     return f'{(query + '\n') * self.copy_count}\n{resp}'

