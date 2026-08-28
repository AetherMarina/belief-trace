from typing import Protocol
import ollama


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


class OllamaEmbedder:
    def __init__(self, model: str = "mxbai-embed-large"):
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = ollama.embed(
            model=self.model,
            input=text,
        )
        return response["embeddings"][0]
