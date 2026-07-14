from rag.document_loader import DocumentLoader
from rag.text_splitter import TextSplitter
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


def build_vector_database():

    print("Loading agriculture documents...")

    loader = DocumentLoader()

    documents = loader.load_all_documents()

    if not documents:

        print(
            "No PDF or TXT documents found "
            "inside data/knowledge_base/"
        )

        return

    print(
        f"Loaded {len(documents)} document pages/files."
    )


    print("Splitting documents...")

    splitter = TextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    print("Generating embeddings...")

    embedding_model = EmbeddingModel()

    texts = [

        chunk["text"]

        for chunk in chunks
    ]

    embeddings = (
        embedding_model.encode_documents(
            texts
        )
    )


    print("Creating FAISS index...")

    vector_store = VectorStore()

    vector_store.create_index(
        embeddings=embeddings,
        documents=chunks,
    )


    print("Saving FAISS index...")

    vector_store.save()


    print(
        "Agriculture vector database "
        "created successfully."
    )


if __name__ == "__main__":

    build_vector_database()