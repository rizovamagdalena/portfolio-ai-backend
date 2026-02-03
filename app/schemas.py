from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class ProjectInfo(BaseModel):
    project_id: str
    project_name: str
    chunk_type: str
    relevance_score: float
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[ProjectInfo]
    projects_searched: List[str]

class HealthResponse(BaseModel):
    status: str
    total_documents: int
    available_projects: dict