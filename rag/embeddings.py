import numpy as np

from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
    ):

        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )


    def encode_documents(self, texts):

        if not texts:

            return np.empty((0, 0), dtype="float32")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        return embeddings.astype("float32")


    def encode_query(self, query):

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype("float32")