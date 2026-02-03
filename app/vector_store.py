
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "projects_collection"):
        # Load OpenAI key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Set your OPENAI_API_KEY in .env!")

        # Initialize LangChain embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.openai_api_key
        )

        # Initialize LangChain Chroma vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )

        # DEBUG: print how many documents exist
        self.print_collection_stats()

    def print_collection_stats(self):
        try:
            # Get collection from the underlying ChromaDB client
            collection = self.vector_store._collection
            count = collection.count()
            print(f"DEBUG: Collection has {count} documents")
        except Exception as e:
            print("DEBUG: Could not retrieve collection stats:", e)

    def add_document(self, doc_id: str, text: str, metadata: dict):
        from langchain_core.documents import Document

        # Add doc_id to metadata
        metadata["doc_id"] = doc_id

        # Create LangChain Document
        doc = Document(page_content=text, metadata=metadata)

        # Add to vector store
        self.vector_store.add_documents([doc])

    def query(self, query_text: str, top_k: int = 3, debug: bool = False):
        results = self.vector_store.similarity_search_with_score(
            query_text,
            k=top_k
        )

        if debug:
            print("DEBUG: LangChain query results:", results)

        # Extract documents, metadatas, and distances
        documents = [doc.page_content for doc, score in results]
        metadatas = [doc.metadata for doc, score in results]
        distances = [score for doc, score in results]

        return documents, metadatas, distances

    def get_documents_only(self, query_text: str, top_k: int = 3):
        docs, _, _ = self.query(query_text, top_k)
        return docs

    def get_relevant_projects(self, query_text: str, top_k: int = 3):
        _, metadatas, _ = self.query(query_text, top_k)
        project_names = set()
        for metadata in metadatas:
            project_names.add(metadata['project_name'])
        return list(project_names)

    def list_all_projects(self):
        try:
            # Access ChromaDB collection
            collection = self.vector_store._collection
            all_docs = collection.get(include=["metadatas"])

            projects = {}
            for metadata in all_docs['metadatas']:
                project_id = metadata['project_id']
                project_name = metadata['project_name']
                if project_id not in projects:
                    projects[project_id] = project_name
            return projects
        except Exception as e:
            print("Error listing projects:", e)
            return {}

    def as_retriever(self, search_kwargs: dict = None):
        if search_kwargs is None:
            search_kwargs = {"k": 3}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)