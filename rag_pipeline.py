from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TOP_K
)

from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore
from prompts import build_prompt


class HRPolicyRAG:

    def __init__(self):

        self.client = Groq(
            api_key=GROQ_API_KEY
        )

        self.embedding_model = EmbeddingModel()

        self.vector_store = FAISSVectorStore()

        self.ready = False

    def build_knowledge_base(self, chunks):

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode_documents(
            texts
        )

        self.vector_store.create_index(
            embeddings,
            chunks
        )

        self.ready = True

    def retrieve(self, question):

        query_embedding = (
            self.embedding_model.encode_query(
                question
            )
        )

        results = self.vector_store.search(
            query_embedding,
            TOP_K
        )

        return results

    def generate_answer(self, question):

        if not self.ready:

            raise ValueError(
                "Knowledge base has not been created."
            )

        documents = self.retrieve(question)

        prompt = build_prompt(
            question,
            documents
        )

        response = self.client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise HR policy assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.1,

            reasoning_effort="low"
        )

        answer = response.choices[0].message.content

        return answer, documents
