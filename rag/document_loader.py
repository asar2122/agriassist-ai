from pathlib import Path

import fitz


class DocumentLoader:

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".txt",
    }

    def __init__(
        self,
        knowledge_base_path="data/knowledge_base",
    ):
        self.knowledge_base_path = Path(
            knowledge_base_path
        )


    def load_pdf(self, file_path):

        documents = []

        pdf_document = fitz.open(file_path)

        try:

            for page_number, page in enumerate(
                pdf_document
            ):

                text = page.get_text(
                    "text"
                ).strip()

                if text:

                    documents.append(
                        {
                            "text": text,
                            "source": str(file_path),
                            "file_name": file_path.name,
                            "page": page_number + 1,
                        }
                    )

        finally:

            pdf_document.close()

        return documents


    def load_text_file(self, file_path):

        try:

            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

        except Exception as error:

            print(
                f"ERROR READING FILE: "
                f"{file_path}"
            )

            print(
                f"ERROR MESSAGE: {error}"
            )

            return []


        if not text:

            print(
                f"WARNING: EMPTY FILE: "
                f"{file_path}"
            )

            return []


        return [
            {
                "text": text,
                "source": str(file_path),
                "file_name": file_path.name,
                "page": None,
            }
        ]


    def load_document(self, file_path):

        extension = file_path.suffix.lower()


        if extension == ".pdf":

            return self.load_pdf(
                file_path
            )


        if extension == ".txt":

            return self.load_text_file(
                file_path
            )


        return []


    def load_all_documents(self):

        print()
        print("=" * 60)
        print("DOCUMENT LOADER DEBUG INFORMATION")
        print("=" * 60)


        print(
            "KNOWLEDGE BASE PATH:"
        )

        print(
            self.knowledge_base_path
        )


        print()
        print(
            "ABSOLUTE KNOWLEDGE BASE PATH:"
        )

        print(
            self.knowledge_base_path.resolve()
        )


        print()
        print(
            "PATH EXISTS:"
        )

        print(
            self.knowledge_base_path.exists()
        )


        if not self.knowledge_base_path.exists():

            raise FileNotFoundError(
                f"Knowledge base directory "
                f"not found: "
                f"{self.knowledge_base_path.resolve()}"
            )


        all_documents = []


        print()
        print("=" * 60)
        print("SEARCHING FOR DOCUMENTS")
        print("=" * 60)


        for file_path in (
            self.knowledge_base_path.rglob("*")
        ):

            print()
            print(
                f"FOUND PATH: {file_path}"
            )


            if file_path.is_dir():

                print(
                    "TYPE: DIRECTORY"
                )

                continue


            print(
                "TYPE: FILE"
            )


            print(
                f"FILE NAME: "
                f"{file_path.name}"
            )


            print(
                f"EXTENSION: "
                f"{file_path.suffix}"
            )


            print(
                f"FILE SIZE: "
                f"{file_path.stat().st_size} bytes"
            )


            if (
                file_path.suffix.lower()
                not in self.SUPPORTED_EXTENSIONS
            ):

                print(
                    "STATUS: UNSUPPORTED FILE"
                )

                continue


            print(
                "STATUS: SUPPORTED DOCUMENT"
            )


            print(
                f"LOADING: {file_path}"
            )


            documents = self.load_document(
                file_path
            )


            print(
                f"DOCUMENTS RETURNED: "
                f"{len(documents)}"
            )


            all_documents.extend(
                documents
            )


        print()
        print("=" * 60)
        print("DOCUMENT LOADING COMPLETE")
        print("=" * 60)


        print(
            f"TOTAL DOCUMENTS LOADED: "
            f"{len(all_documents)}"
        )


        print()


        if all_documents:

            print(
                "LOADED DOCUMENTS:"
            )


            for number, document in enumerate(
                all_documents,
                start=1,
            ):

                print(
                    f"{number}. "
                    f"{document['file_name']}"
                )


        else:

            print(
                "WARNING: NO DOCUMENTS LOADED"
            )


        print()
        print("=" * 60)


        return all_documents