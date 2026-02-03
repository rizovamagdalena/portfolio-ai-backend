
import os
import chromadb
from openai import OpenAI


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "projects_collection"):
        # Load OpenAI key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("Set your OPENAI_API_KEY in .env!")

        # OpenAI client
        self.client = OpenAI(api_key=self.openai_api_key)

        # ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)

        # Get collection
        self.collection = self.chroma_client.get_or_create_collection(collection_name)

        # DEBUG: print how many documents exist
        self.print_collection_stats()

    def print_collection_stats(self):
        try:
            all_docs = self.collection.get(include=["documents", "metadatas"])
            print(f"DEBUG: Collection has {len(all_docs['ids'])} documents")
        except Exception as e:
            print("DEBUG: Could not retrieve collection stats:", e)

    def add_document(self, doc_id: str, text: str, metadata: dict):
        # Create embedding
        embedding = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        ).data[0].embedding

        # Add to ChromaDB
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text]
        )

    def query(self, query_text: str, top_k: int = 3, debug: bool = False):

        # Generate query embedding
        query_embedding = self.client.embeddings.create(
            input=query_text,
            model="text-embedding-3-large"
        ).data[0].embedding

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if debug:
            print("DEBUG: Chroma query results:", results)

        # Return documents, metadatas, and distances
        return results['documents'][0], results['metadatas'][0], results['distances'][0]

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
            all_docs = self.collection.get(include=["metadatas"])
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