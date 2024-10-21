import re
from typing import Dict


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

# Reciprocal Rank Fusion algorithm
# https://github.com/Raudaschl/rag-fusion/blob/master/main.py
def reciprocal_rank_fusion(search_results_dict: Dict[str, Dict[str, float]], k=60):
    fused_scores = {}
    # print("Initial individual search result ranks:")
    # for query, doc_scores in search_results_dict.items():
    #     print(f"For query '{query}': {doc_scores}")
        
    for query, doc_scores in search_results_dict.items():
        for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            # previous_score = fused_scores[doc]
            fused_scores[doc] += 1 / (rank + k)
            # print(f"Updating score for {doc} from {previous_score} to {fused_scores[doc]} based on rank {rank} in query '{query}'")

    reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    # print("Final reranked results:", reranked_results)
    return reranked_results
