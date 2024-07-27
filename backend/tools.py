import re


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

def get_embedding_model(using_custom_embedding_model, custom_embedding_model_name, tokenizer) -> str:
    return custom_embedding_model_name if using_custom_embedding_model else tokenizer.model_name + '-' + tokenizer.embedding_model
