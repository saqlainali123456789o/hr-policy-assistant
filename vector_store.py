import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(self):

        self.index = None
        self.documents = []

    def create_index(self, embeddings, documents):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.documents = documents

    def search(self, query_embedding, top_k=5):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            document = self.documents[index].copy()

            document["score"] = float(score)

            results.append(document)

        return results
