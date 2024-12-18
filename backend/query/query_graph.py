import tiktoken
from typing import Dict, Tuple, List
from .graph_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
from settings.settings import Settings


class GraphSeach:
    def __init__(
        self, 
        llm,
        embeddings,
        vectorstore,
        db,
        settings: Settings,
        system_prompt: str = LOCAL_SEARCH_SYSTEM_PROMPT,
        response_type: str = "multiple paragraphs",
    ) -> None:
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.db = db
        self.settings = settings
        self.system_prompt = system_prompt
        self.response_type = response_type
    
    def search(self, query: str, file_id: str, **kw):
        entities = map_query_to_entities(self.vectorstore, self.embeddings, self.db, query, file_id, self.settings.embeddings.dim)
        # print(entities)
        entity_context, entities = build_entity_context(entities)
        # print(entity_context)
        relationships = _filter_relationships(self.db, entities, file_id)
        relationship_context, relationships = build_relationship_context(relationships)
        # print(relationship_context)
        context_text = entity_context + "\n\n" + relationship_context
        search_prompt = self.system_prompt.format(
            context_data=context_text, response_type=self.response_type
        )
        resp = self.llm.chat(query, search_prompt)
        # print(resp)
        return resp


def map_query_to_entities(
    vectorstore, embeddings, db, 
    query: str, file_id: str, dim: int
):
    query_emb = embeddings.encode(query)
    result = vectorstore.search_data(
        dim, query_emb, 60, 
        filter=f"file_id == '{file_id}'")
    entities = vectorstore.get_data(dim, [r['id'] for r in result], output_fields=['file_id', 'entity_id'])
    # for e in entities:
    #     print(e)
    return db.get_data(
        'entities', 
        # {'file_id': {'$in': [e['file_id'] for e in entities]}},
        {
            '$and': [
                {'file_id': file_id},
                # {'file_id': {'$in': [e['file_id'] for e in entities]}},
                {'id': {'$in': [e['entity_id'] for e in entities if 'entity_id' in e]}},
                # {'$not': [{'description': ''}]},
            ]
        },
        
    )

def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
    """Return the number of tokens in the given text."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    return len(token_encoder.encode(text))

def build_entity_context(
    selected_entities: Dict,
    token_encoder: tiktoken.Encoding | None = None,
    max_tokens: int = 4000,
    # include_entity_rank: bool = True,
    # rank_description: str = "number of relationships",
    column_delimiter: str = "|",
    context_name="Entities",
) -> Tuple[str, List]:
    """Prepare entity data table as context data for system prompt."""
    if len(selected_entities) == 0:
        return "", None
    entities = []

    # add headers
    current_context_text = f"-----{context_name}-----" + "\n"
    header = ["id", "entity", "description"]
    # if include_entity_rank:
    #     header.append(rank_description)
    # attribute_cols = (
    #     list(selected_entities[0].attributes.keys())
    #     if selected_entities[0].attributes
    #     else []
    # )
    # header.extend(attribute_cols)
    current_context_text += column_delimiter.join(header) + "\n"
    current_tokens = num_tokens(current_context_text, token_encoder)

    # all_context_records = [header]
    for i, entity in enumerate(selected_entities):
        new_context = [
            str(i),
            entity['name'].lower(),
            entity['description'] if entity['description'] else "",
        ]
        # if include_entity_rank:
        #     new_context.append(str(entity.rank))
        # for field in attribute_cols:
        #     field_value = (
        #         str(entity.attributes.get(field))
        #         if entity.attributes and entity.attributes.get(field)
        #         else ""
        #     )
        #     new_context.append(field_value)
        new_context_text = column_delimiter.join(new_context) + "\n"
        new_tokens = num_tokens(new_context_text, token_encoder)
        if current_tokens + new_tokens > max_tokens:
            break
        current_context_text += new_context_text
        # all_context_records.append(new_context)
        current_tokens += new_tokens
        entities.append(entity)

    return current_context_text, entities

def _filter_relationships(
    db,
    selected_entities: Dict,
    file_id: str,
) -> Dict:
    return db.get_data(
        'relationships', 
        {'$and': [
            {'file_id': file_id}, 
            {'$or': [
                {'source': {'$in': [e['name'] for e in selected_entities]}},
                {'target': {'$in': [e['name'] for e in selected_entities]}}
            ]},
            {'$nor': [{'description': ''}]}
        ]}
    )

def build_relationship_context(
    selected_relationships: Dict,
    token_encoder: tiktoken.Encoding | None = None,
    include_relationship_weight: bool = False,
    max_tokens: int = 4000,
    # top_k_relationships: int = 10,
    # relationship_ranking_attribute: str = "rank",
    column_delimiter: str = "|",
    context_name: str = "Relationships",
) -> tuple[str, ]:
    """Prepare relationship data tables as context data for system prompt."""
    if len(selected_relationships) == 0 or len(selected_relationships) == 0:
        return "", None
    relationships = []

    # add headers
    current_context_text = f"-----{context_name}-----" + "\n"
    header = ["id", "source", "target", "description"]
    if include_relationship_weight:
        header.append("weight")
    # attribute_cols = (
    #     list(selected_relationships[0].attributes.keys())
    #     if selected_relationships[0].attributes
    #     else []
    # )
    # attribute_cols = [col for col in attribute_cols if col not in header]
    # header.extend(attribute_cols)

    current_context_text += column_delimiter.join(header) + "\n"
    current_tokens = num_tokens(current_context_text, token_encoder)

    # all_context_records = [header]
    for i, rel in enumerate(selected_relationships):
        new_context = [
            str(i),
            rel['source'],
            rel['target'],
            rel['description'] if rel['description'] else "",
        ]
        if include_relationship_weight:
            new_context.append(str(rel['weight'] if ['weight'] else ""))
        # for field in attribute_cols:
        #     field_value = (
        #         str(rel.attributes.get(field))
        #         if rel.attributes and rel.attributes.get(field)
        #         else ""
        #     )
        #     new_context.append(field_value)
        new_context_text = column_delimiter.join(new_context) + "\n"
        new_tokens = num_tokens(new_context_text, token_encoder)
        if current_tokens + new_tokens > max_tokens:
            break
        current_context_text += new_context_text
        # all_context_records.append(new_context)
        current_tokens += new_tokens
        relationships.append(rel)

    return current_context_text, relationships
