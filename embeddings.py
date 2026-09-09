from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def encode_documents(self, texts):

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def encode_query(self, query):

        return self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
