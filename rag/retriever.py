from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


class AgricultureRetriever:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        index_directory="vector_database/faiss_index",
    ):

        self.embedding_model = EmbeddingModel(
            model_name=model_name
        )

        self.vector_store = VectorStore(
            index_directory=index_directory
        )

        self.vector_store.load()


    def retrieve(
        self,
        question,
        top_k=5,
        minimum_score=0.25,
    ):

        if not question.strip():

            return []

        query_embedding = (
            self.embedding_model.encode_query(
                question
            )
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        filtered_results = [

            result

            for result in results

            if result["score"] >= minimum_score
        ]

        return filtered_results


    def get_context(
        self,
        question,
        top_k=5,
    ):

        results = self.retrieve(
            question=question,
            top_k=top_k,
        )

        context_parts = []

        for result in results:

            source = result["file_name"]

            page = result.get("page")

            if page:

                source_information = (
                    f"Source: {source}, Page: {page}"
                )

            else:

                source_information = (
                    f"Source: {source}"
                )

            context_parts.append(
                f"{source_information}\n"
                f"{result['text']}"
            )

        return "\n\n".join(context_parts)