from injector import inject, singleton
from settings.settings import Settings
from embeddings import Embeddings, LocalEmbeddings, OpenaiEmbeddings


class EmbeddingsComponentBase:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        model = settings.embeddings.model
        match settings.embeddings.mode:
            case 'local':
                self.embeddings = LocalEmbeddings(model)
            case 'openai':
                self.embeddings = OpenaiEmbeddings(model, api_base=settings.embeddings.api_base)
            case 'mock':
                self.embeddings = Embeddings()

@singleton
class EmbeddingsComponent(EmbeddingsComponentBase):
    @inject
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
