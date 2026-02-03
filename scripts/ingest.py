import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.config import Settings
import chromadb

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
# Init the OPENAI client
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Init chroma db
# -----------------------------
# chroma_client = chromadb.Client(Settings(
#     persist_directory=CHROMA_PERSIST_DIR
# ))

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


collection_name = "projects_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)
print(f"Using collection: {collection_name}")

# -----------------------------
# Load projects.json file
# -----------------------------
with open(PROJECTS_JSON_PATH, "r", encoding="utf-8") as f:
    projects = json.load(f)

print(f"Loaded {len(projects)} projects from {PROJECTS_JSON_PATH}")

# -----------------------------
# Ingest
# -----------------------------
for project in projects:
    project_id = project["id"]
    project_name = project["name"]
    chunks = project.get("chunks", [])

    print(f"\n➡️ Processing project: {project_name} ({project_id}) with {len(chunks)} chunks")

    for idx, chunk in enumerate(chunks):
        text = chunk["text"]
        chunk_type = chunk.get("type", "General")

        print(f"  - Generating embedding for chunk {idx} [{chunk_type}]")

        # Generate embedding
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding

        # Add to ChromaDB
        collection.add(
            ids=[f"{project_id}_{idx}"],
            embeddings=[embedding],
            metadatas=[{
                "project_id": project_id,
                "project_name": project_name,
                "chunk_type": chunk_type
            }],
            documents=[text]
        )

        print(f"    ✅ Chunk {idx} added to collection")
        # -----------------------------
        # Debug: print total docs in collection
        # -----------------------------
        try:
            all_docs = collection.get(include=["documents", "metadatas"])
            print(f"    🔹 Collection now has {len(all_docs['documents'])} documents")
            print(f"    🔹 Last added doc metadata: {all_docs['metadatas'][-1]}")
            print(f"    🔹 Last added doc text: {all_docs['documents'][-1]}\n")
        except Exception as e:
            print("    ⚠️ Could not retrieve collection stats:", e)

print("Chroma DB path:", CHROMA_PERSIST_DIR)

print("\n🎉 Ingestion complete! ChromaDB is now populated.")
