from typing import Union
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

    def process_query(self, query: str) -> list[str]:
        raise NotImplementedError('Not implemented')


    def _process_query(self, query: str) -> str:
        resp = self.llm.chat(self.prompt_tmpl.format(query=query))
        return resp


class QueryOptimiserHyDE(QueryOptimizer):
    '''
    首先，HyDE 针对 query 直接生成一个假设性文档或者说回答（hypo_doc）。
    然后，对这个假设性回答进行向量化处理。最后，使用向量化的假设性回答去检索相似文档。

    模仿 llamaindex 的 HyDEQueryTransform 类。
    '''
    name = 'hyde'
    prompt_tmpl = (
        "Please write a passage to answer the question\n"
        "Try to include as many key details as possible.\n"
        "\n"
        "\n"
        "{query}\n"
        "\n"
        "\n"
        'Passage:"""\n'
    )

    def process_query(self, query: str) -> list[str]:
        return [self._process_query(query)]


class QueryOptimiserMultiQuery(QueryOptimizer):
    '''
    通过生成多种视角的查询来检索相关文档。

    模仿 langchain 的 MultiQueryRetriever 类。
    '''
    name ='multi_query'
    prompt_tmpl = """You are an AI language model assistant. Your task is 
    to generate 3 different versions of the given user 
    question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by newlines. Original question: {query}"""

    def process_query(self, query: str) -> list[str]:
        resp = self._process_query(query)
        lines = resp.strip().split("\n")
        return lines


# class QueryOptimiserStepBackPrompting(QueryOptimizer):
#     name ='step_back_prompting'


# class QueryOptimiserRagFusion(QueryOptimizer):
#     '''
#     https://github.com/Raudaschl/rag-fusion/blob/master/main.py

#     核心原理是根据用户的原始 query 生成多个不同角度的 query ，以捕捉 query 的不同方面和细微差别。
#     然后通过使用逆向排名融合（Reciprocal Rank Fusion，RRF）技术，将多个 query 的检索结果进行融合，
#     生成一个统一的排名列表，从而增加最相关文档出现在最终 TopK 列表的机会。
#     '''
#     name = 'rag_fusion'

#     def generate_queries_chatgpt(self, original_query):
#         response = self.llm.llm.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
#                 {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
#                 {"role": "user", "content": "OUTPUT (4 queries):"}
#             ]
#         )

#         generated_queries = response.choices[0]["message"]["content"].strip().split("\n")
#         return generated_queries
    
#     @staticmethod
#     def reciprocal_rank_fusion(search_results_dict, k=60):
#         fused_scores = {}
#         print("Initial individual search result ranks:")
#         for query, doc_scores in search_results_dict.items():
#             print(f"For query '{query}': {doc_scores}")
            
#         for query, doc_scores in search_results_dict.items():
#             for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
#                 if doc not in fused_scores:
#                     fused_scores[doc] = 0
#                 previous_score = fused_scores[doc]
#                 fused_scores[doc] += 1 / (rank + k)
#                 print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

#         reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
#         print("Final reranked results:", reranked_results)
#         return reranked_results

#     def process_query(self, query: str):
#         return query

