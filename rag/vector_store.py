import json

from pathlib import Path

import faiss
import numpy as np


class VectorStore:

    def __init__(
        self,
        index_directory="vector_database/faiss_index",
    ):

        self.index_directory = Path(
            index_directory
        )

        self.index_path = (
            self.index_directory / "agriassist.index"
        )

        self.metadata_path = (
            self.index_directory / "metadata.json"
        )

        self.index = None

        self.metadata = []


    def create_index(
        self,
        embeddings,
        documents,
    ):

        if embeddings.size == 0:

            raise ValueError(
                "Cannot create FAISS index "
                "from empty embeddings."
            )

        if len(embeddings) != len(documents):

            raise ValueError(
                "Embeddings and documents "
                "must have the same length."
            )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        self.index.add(embeddings)

        self.metadata = documents


    def save(self):

        if self.index is None:

            raise ValueError(
                "No FAISS index available to save."
            )

        self.index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )


    def load(self):

        if not self.index_path.exists():

            raise FileNotFoundError(
                "FAISS index does not exist. "
                "Run build_vector_database.py first."
            )

        if not self.metadata_path.exists():

            raise FileNotFoundError(
                "FAISS metadata does not exist."
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            self.metadata = json.load(file)


    def search(
        self,
        query_embedding,
        top_k=5,
    ):

        if self.index is None:

            raise ValueError(
                "FAISS index has not been loaded."
            )

        if top_k <= 0:

            return []

        available_documents = len(
            self.metadata
        )

        if available_documents == 0:

            return []

        top_k = min(
            top_k,
            available_documents,
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index == -1:

                continue

            document = self.metadata[index].copy()

            document["score"] = float(score)

            results.append(document)

        return results