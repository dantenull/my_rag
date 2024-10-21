import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from span_marker import SpanMarkerModel
from typing import Dict, List

DEFAULT_ENTITY_MAP = {
    "PER": "persons",
    "ORG": "organizations",
    "LOC": "locations",
    "ANIM": "animals",
    "BIO": "biological",
    "CEL": "celestial",
    "DIS": "diseases",
    "EVE": "events",
    "FOOD": "foods",
    "INST": "instruments",
    "MEDIA": "media",
    "PLANT": "plants",
    "MYTH": "mythological",
    "TIME": "times",
    "VEHI": "vehicles",
}


def ner(text: str, entity_types: list[str]) -> Dict[List[str]]:
    # connected_entities = []
    entity_map = {}
    a = nltk.pos_tag(nltk.word_tokenize(text))
    for chunk in nltk.ne_chunk(a):
        if hasattr(chunk, "label"):
            entity_type = chunk.label().lower()
            if entity_type not in entity_types:
                continue
            name = (" ".join(c[0] for c in chunk)).upper()
            # connected_entities.append(name)
            if name not in entity_map:
                entity_map[name] = entity_type

    metadata = {}
    for span, lable in entity_map.items():
        # TODO DEFAULT_ENTITY_MAP 可能不能跟下面的方法用同一个
        metadata_label = DEFAULT_ENTITY_MAP.get(lable, lable)
        if metadata_label not in metadata:
            metadata[metadata_label] = set()
        metadata[metadata_label].add(span)

    for key, val in metadata.items():
        metadata[key] = list(val)

    return entity_map


def ner_with_model(
        model_name: str,
        text: str, 
        prediction_threshold: float = 0.5,
        span_joiner = ' ',
    ) -> Dict[List[str]]:
    model = SpanMarkerModel.from_pretrained(model_name)
    words = word_tokenize(text)
    spans = model.predict(words)

    metadata = {}
    for span in spans:
        if span["score"] < prediction_threshold:
            continue
        metadata_label = DEFAULT_ENTITY_MAP.get(span["label"], span["label"])
        if metadata_label not in metadata:
            metadata[metadata_label] = set()
        metadata[metadata_label].add(span_joiner.join(span["span"]))

    for key, val in metadata.items():
        metadata[key] = list(val)
    return metadata

