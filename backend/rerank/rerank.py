from sentence_transformers import CrossEncoder
import numpy as np
from typing import List


class Rerank:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def rerank(self, query: str, texts: List[str]):
        cross_encoder = CrossEncoder(self.model_name)
        pairs = [[query, doc] for doc in texts]
        print(pairs)
        scores = cross_encoder.predict(pairs)
        result = []
        for o in np.argsort(scores)[::-1]:
            result.append(texts[o])
        return result

