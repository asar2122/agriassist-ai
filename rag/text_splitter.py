class TextSplitter:

    def __init__(
        self,
        chunk_size=500,
        chunk_overlap=100,
    ):

        if chunk_size <= 0:

            raise ValueError(
                "chunk_size must be greater than zero."
            )

        if chunk_overlap < 0:

            raise ValueError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:

            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap


    def split_text(self, text):

        words = text.split()

        chunks = []

        start = 0

        while start < len(words):

            end = start + self.chunk_size

            chunk_words = words[start:end]

            chunk = " ".join(chunk_words).strip()

            if chunk:

                chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks


    def split_documents(self, documents):

        chunked_documents = []

        chunk_id = 0

        for document in documents:

            chunks = self.split_text(
                document["text"]
            )

            for chunk_number, chunk in enumerate(chunks):

                chunked_documents.append(
                    {
                        "id": chunk_id,

                        "text": chunk,

                        "source": document["source"],

                        "file_name": document["file_name"],

                        "page": document["page"],

                        "chunk_number": chunk_number,
                    }
                )

                chunk_id += 1

        return chunked_documents