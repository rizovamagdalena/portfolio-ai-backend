
import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# -----------------------------
# Load .env variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY in the .env file!")

PROJECTS_JSON_PATH = "../data/projects.json"
CHROMA_PERSIST_DIR = os.path.abspath("../data/chroma_db")
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# -----------------------------
# Init embeddings with LangChain
# -----------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=OPENAI_API_KEY
)

# -----------------------------
# Init Chroma with LangChain
# -----------------------------
vector_store = Chroma(
    collection_name="projects_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIR
)

print(f"Using collection: projects_collection")

# -----------------------------
# Load projects.json file
# -----------------------------
with open(PROJECTS_JSON_PATH, "r", encoding="utf-8") as f:
    projects = json.load(f)

print(f"Loaded {len(projects)} projects from {PROJECTS_JSON_PATH}")

# -----------------------------
# Ingest
# -----------------------------
all_documents = []

for project in projects:
    project_id = project["id"]
    project_name = project["name"]
    chunks = project.get("chunks", [])

    print(f"\n➡️ Processing project: {project_name} ({project_id}) with {len(chunks)} chunks")

    for idx, chunk in enumerate(chunks):
        text = chunk["text"]
        chunk_type = chunk.get("type", "General")

        print(f"  - Preparing chunk {idx} [{chunk_type}]")

        # Create LangChain Document
        doc = Document(
            page_content=text,
            metadata={
                "project_id": project_id,
                "project_name": project_name,
                "chunk_type": chunk_type,
                "doc_id": f"{project_id}_{idx}"
            }
        )

        all_documents.append(doc)
        print(f"    ✅ Chunk {idx} prepared")

# Add all documents to vector store in batch
print(f"\n📦 Adding {len(all_documents)} documents to vector store...")
vector_store.add_documents(all_documents)

# -----------------------------
# Debug: print total docs in collection
# -----------------------------
print(f"\n🔹 Collection now has documents stored in: {CHROMA_PERSIST_DIR}")

print("\n🎉 Ingestion complete! ChromaDB is now populated with LangChain.")