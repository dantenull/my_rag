from pathlib import Path

PROJECT_ROOT_PATH: Path = Path(__file__).parents[0]

DEFAULT_CHUNK_SIZE = 512
SENTENCE_CHUNK_OVERLAP = 200
CHUNKING_REGEX = "[^,.;?!，。；？！]+[,.;?!，。；？！]?"
DEFAULT_PARAGRAPH_SEP = "\n"
