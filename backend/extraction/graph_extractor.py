from typing import Any, List, Dict
from .prompts import GRAPH_EXTRACTION_PROMPT, CONTINUE_PROMPT, LOOP_PROMPT
from llms import LLM
import re
import html
from collections.abc import Mapping
import networkx as nx
# from networkx.readwrite import json_graph


DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


class GraphExtractor:
    def __init__(
        self, 
        llm: LLM,
    ) -> None:
        self._max_gleanings = 3
        self.llm = llm
        self._join_descriptions = True
        # self.extraction_prompt = GRAPH_EXTRACTION_PROMPT.format(
        #     entity_types=DEFAULT_ENTITY_TYPES, 
        #     tuple_delimiter=DEFAULT_TUPLE_DELIMITER, 
        #     record_delimiter=DEFAULT_RECORD_DELIMITER, 
        #     completion_delimiter=DEFAULT_COMPLETION_DELIMITER)
    
    def __call__(self, texts: list[str]) -> Any:
        source_doc_map = {}
        results = {}
        for doc_index, text in enumerate(texts):
            source_doc_map[doc_index] = text
            result = self._process_documents(text)
            results[doc_index] = result
        graph = self._process_results(results)
        # output = json_graph.tree_data(graph, root=1)
        return source_doc_map, graph

    def _process_documents(self, text: str):
        query = GRAPH_EXTRACTION_PROMPT.format(
            entity_types=DEFAULT_ENTITY_TYPES, 
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER, 
            record_delimiter=DEFAULT_RECORD_DELIMITER, 
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            input_text=text)
        results = self.llm.chat(query=query)
        history = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": results},
        ]
        for i in range(self._max_gleanings):
            # TODO 以下两个prompt没有history，手动修改promtp加上
            response = self.llm.chat(CONTINUE_PROMPT, history=history)
            results += response or ""
            history = [
                *history,
                {"role": "user", "content": CONTINUE_PROMPT},
                {"role": "assistant", "content": response},
            ]

            if i >= self._max_gleanings - 1:
                break

            response = self.llm.chat(
                LOOP_PROMPT,
                history=history
            )
            history = [
                *history,
                {"role": "user", "content": LOOP_PROMPT},
                {"role": "assistant", "content": response},
            ]
            if response != "YES":
                break
        return results
    
    def _process_results(self, results: Dict):
        graph = nx.Graph()
        for source_doc_id, extracted_data in results.items():
            records = [r.strip() for r in extracted_data.split(DEFAULT_RECORD_DELIMITER)]
            # print(records)
            for record in records:
                record = re.sub(r"^\(|\)$", "", record.strip())
                record_attributes = record.split(DEFAULT_TUPLE_DELIMITER)

                if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                    # add this record as a node in the G
                    entity_name = clean_str(record_attributes[1].upper())
                    entity_type = clean_str(record_attributes[2].upper())
                    entity_description = clean_str(record_attributes[3])

                    if entity_name in graph.nodes():
                        node = graph.nodes[entity_name]
                        if self._join_descriptions:
                            node["description"] = "\n".join(
                                list({
                                    *_unpack_descriptions(node),
                                    entity_description,
                                })
                            )
                        else:
                            if len(entity_description) > len(node["description"]):
                                node["description"] = entity_description
                        node["source_id"] = ", ".join(
                            list({
                                *_unpack_source_ids(node),
                                str(source_doc_id),
                            })
                        )
                        node["entity_type"] = (
                            entity_type if entity_type != "" else node["entity_type"]
                        )
                    else:
                        graph.add_node(
                            entity_name,
                            type=entity_type,
                            description=entity_description,
                            source_id=str(source_doc_id),
                        )

                if (
                    record_attributes[0] == '"relationship"'
                    and len(record_attributes) >= 5
                ):
                    # add this record as edge
                    source = clean_str(record_attributes[1].upper())
                    target = clean_str(record_attributes[2].upper())
                    edge_description = clean_str(record_attributes[3])
                    edge_source_id = clean_str(str(source_doc_id))
                    try:
                        weight = float(record_attributes[-1])
                    except ValueError:
                        weight = 1.0

                    if source not in graph.nodes():
                        graph.add_node(
                            source,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if target not in graph.nodes():
                        graph.add_node(
                            target,
                            type="",
                            description="",
                            source_id=edge_source_id,
                        )
                    if graph.has_edge(source, target):
                        edge_data = graph.get_edge_data(source, target)
                        if edge_data is not None:
                            weight += edge_data["weight"]
                            if self._join_descriptions:
                                edge_description = "\n".join(
                                    list({
                                        *_unpack_descriptions(edge_data),
                                        edge_description,
                                    })
                                )
                            edge_source_id = ", ".join(
                                list({
                                    *_unpack_source_ids(edge_data),
                                    str(source_doc_id),
                                })
                            )
                    graph.add_edge(
                        source,
                        target,
                        weight=weight,
                        description=edge_description,
                        source_id=edge_source_id,
                    )
        return graph


def _unpack_descriptions(data: Mapping) -> list[str]:
    value = data.get("description", None)
    return [] if value is None else value.split("\n")


def _unpack_source_ids(data: Mapping) -> list[str]:
    value = data.get("source_id", None)
    return [] if value is None else value.split(", ")
