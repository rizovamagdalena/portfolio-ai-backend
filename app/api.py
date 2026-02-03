from fastapi import APIRouter, HTTPException
from app.schemas import QueryRequest, QueryResponse, HealthResponse, ProjectInfo
from app.vector_store import VectorStore
from app.llm import LLM
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

_vector_store = None
_llm = None


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
        _vector_store = VectorStore(persist_dir=CHROMA_PERSIST_DIR)
    return _vector_store


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        vector_store = get_vector_store()
        all_projects = vector_store.list_all_projects()
        all_docs = vector_store.collection.get(include=["documents"])

        return HealthResponse(
            status="healthy",
            total_documents=len(all_docs['ids']),
            available_projects=all_projects
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_projects(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        vector_store = get_vector_store()
        llm = get_llm()

        # Retrieve relevant documents
        docs, metadatas, distances = vector_store.query(
            request.query,
            top_k=request.top_k
        )

        if not docs:
            return QueryResponse(
                answer="I don't have information about that in my project database.",
                sources=[],
                projects_searched=[]
            )

        # Get unique project names
        projects_searched = vector_store.get_relevant_projects(request.query, top_k=request.top_k)

        # Generate answer
        answer = llm.ask_with_context(request.query, docs, metadatas=metadatas)

        # Build sources list
        sources = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            sources.append(ProjectInfo(
                project_id=meta['project_id'],
                project_name=meta['project_name'],
                chunk_type=meta['chunk_type'],
                relevance_score=round(1 / dist if dist > 0 else 100, 2),
                content=doc
            ))

        return QueryResponse(
            answer=answer,
            sources=sources,
            projects_searched=projects_searched
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects")
async def list_projects():
    try:
        vector_store = get_vector_store()
        projects = vector_store.list_all_projects()
        return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))