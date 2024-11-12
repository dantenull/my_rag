import re
from typing import Dict, Tuple, Union, Callable, List


def content_title_process(s: str) -> str:
    # 在对LLM进行提问时，由于它无法准确无误的回答一模一样的标题选项，需要对它回答的标题进行处理，
    # 以便在后续的判断时可以匹配数据库中存储的标题
    if not s:
        return s
    result = s.strip().lower()
    pattern = re.compile(r'^[\d\.:,<>\[\]\{\}\+\-\*\\/\(\)\?]*(.+)')
    m= pattern.match(result)
    if m:
        result = m.group(1)
        result = result.strip()
    symbols = ['.', '，', '。', '\n', ' ']
    for symbol in symbols:
        result = result.replace(symbol, '')
    return result

def get_embedding_model(custom_embedding_model, custom_embedding_model_name, tokenizer) -> str:
    using_custom_embedding_model = (custom_embedding_model != '' and custom_embedding_model is not None)
    return custom_embedding_model_name if using_custom_embedding_model else tokenizer.model_name + '-' + tokenizer.embedding_model

# RRF(Reciprocal Rank Fusion algorithm)
# 根据 https://github.com/Raudaschl/rag-fusion/blob/master/main.py 修改
def reciprocal_rank_fusion(search_results_dict: List[List], k: int = 60) -> List[Tuple]:
    fused_scores = {}
    for docs in search_results_dict:
        for rank, doc in enumerate(docs):
            doc_id, text, data = doc
            if doc_id not in fused_scores:
                fused_scores[doc_id] = [0, text, data]
            # previous_score = fused_scores[doc]
            fused_scores[doc_id][0] += 1 / (rank + k)
            # print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = [(doc_id, score[0], score[1], score[2]) for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)]
    # print("Final reranked results:", reranked_results)
    return reranked_results

async def get_chunk_context_async(chat: Callable, chunk: str, whole_document: str, prompt: str=None) -> str:
    if not prompt:
        prompt = '请简要概述指定段落在整个文档中的位置，以便改善对该段落的搜索检索。仅提供简要概述，不要提供其他内容。'
    query = f'以下是文档全文：\n{whole_document}' \
            f'以下是我们想要放置在整个文档中的块：\n{chunk}'
    # GET_CONTEXT_PROMPT = """
    #     以下是全文：

    #     {whole_document}

    #     以下是我们想要放置在整个文档中的块:

    #     {chunk_content}

    #     请简要概述此段落在整个文档中的位置，以便改善对该段落的搜索检索。仅提供简要概述，不要提供其他内容。
    # """
    # if not prompt:
    #     prompt = GET_CONTEXT_PROMPT
    # prompt = prompt.format(whole_document=whole_document, chunk_content=chunk)
    # openai_llm = OpenaiLLM(
    #     'gpt-4o-mini', 
    #     custom_embedding_model=None,
    #     api_base='https://api.gptsapi.net/v1'
    # )
    result = await chat(prompt=prompt, query=query)
    return result

def get_chunk_context(chat: Callable, chunk: str, whole_document: str, prompt: str=None) -> str:
    if not prompt:
        prompt = '请简要概述指定段落在整个文档中的位置，以便改善对该段落的搜索检索。仅提供简要概述，不要提供其他内容。'
    query = f'以下是文档全文：\n{whole_document}' \
            f'以下是我们想要放置在整个文档中的块：\n{chunk}'
    # GET_CONTEXT_PROMPT = """
    #     以下是全文：

    #     {whole_document}

    #     以下是我们想要放置在整个文档中的块:

    #     {chunk_content}

    #     请简要概述此段落在整个文档中的位置，以便改善对该段落的搜索检索。仅提供简要概述，不要提供其他内容。
    # """
    # if not prompt:
    #     prompt = GET_CONTEXT_PROMPT
    # prompt = prompt.format(whole_document=whole_document, chunk_content=chunk)
    # openai_llm = OpenaiLLM(
    #     'gpt-4o-mini', 
    #     custom_embedding_model=None,
    #     api_base='https://api.gptsapi.net/v1'
    # )
    result = chat(prompt=prompt, query=query)
    return result

# def get_start_end_index_around(n: int, total: int, index: int) -> Tuple[int, Union[int, None]]:
#     """
#     TODO
#     """
#     a = (2 * n + 1)
#     if total <= a:
#         return 0, None

#     if index <= n:
#         return 0, a
    
#     if index >= (total -n):
#         return -a, None

#     start = index - 5
#     end = start + a
#     return start, end
