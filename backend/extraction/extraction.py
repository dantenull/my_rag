# from .nltk import ner, ner_with_model
from functools import partial
from typing import Dict, List
from .graph_extractor import GraphExtractor
import networkx as nx

class ExtractionBase:
    def __init__(self, mode: str, **kw) -> None:
        self.extractor = None
        # if mode == 'only_nltk':
        #     # self.extractor = partial(ner, entity_types=kw['entity_types'])
        #     self.extractor = ner
        # elif mode == 'nltk_and_model':
        #     self.extractor = partial(ner_with_model, model_name=kw['model_name'])
        if mode == 'graph':
            self.llm = kw.get('llm')
            if not self.llm:
                raise ValueError('no llm')
            self.extractor = GraphExtractor(self.llm)
    
    def extract_graph(self, docs: List):
        if not self.extractor:
            return
        _, graph = self.extractor(texts=[doc['chunk'] for doc in docs])

        for _, node in graph.nodes(data=True):  # type: ignore
            if node is not None:
                node["source_id"] = ",".join(
                    docs[int(id)]['id'] for id in node["source_id"].split(",")
                )

        for _, _, edge in graph.edges(data=True):  # type: ignore
            if edge is not None:
                edge["source_id"] = ",".join(
                    docs[int(id)]['id'] for id in edge["source_id"].split(",")
                )

        entities = [
            ({"name": item[0], **(item[1] or {})})
            for item in graph.nodes(data=True)
            if item is not None
        ]

        graph_data = "".join(nx.generate_graphml(graph))
        return entities, graph
    