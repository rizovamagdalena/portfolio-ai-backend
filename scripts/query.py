
from dotenv import load_dotenv
from app.vector_store import VectorStore
from app.llm import LLM

# Load environment
load_dotenv()

# Init modules
CHROMA_PERSIST_DIR = "../data/chroma_db"
vector_store = VectorStore(persist_dir=CHROMA_PERSIST_DIR)
llm = LLM()

if __name__ == "__main__":
    print("\n📂 Available projects:")
    all_projects = vector_store.list_all_projects()
    for proj_id, proj_name in all_projects.items():
        print(f"  - {proj_name} ({proj_id})")
    print()

    while True:
        user_query = input("Ask about your projects: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        docs, metadatas, distances = vector_store.query(user_query, top_k=3, debug=True)

        project_names = vector_store.get_relevant_projects(user_query)
        print(f"\n📚 Found information in: {', '.join(project_names)}")

        print(f"\nDEBUG: Retrieved {len(docs)} chunks:")
        for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
            print(f"  [{i + 1}] {meta['project_name']} - {meta['chunk_type']}")
            print(f"      Relevance: {1 / dist:.2f}" if dist > 0 else "      Relevance: Perfect")
            print(f"      Text: {doc[:100]}...\n")

        answer = llm.ask_with_context(user_query, docs, metadatas=metadatas)
        print("\n🤖 Answer:", answer, "\n")
