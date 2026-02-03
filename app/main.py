from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.api import router
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Project Portfolio RAG API",
    description="Query a software engineer's project portfolio using natural language",
    version="1.0.0"
)


# Ultra-permissive CORS for development
@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    # Handle preflight OPTIONS requests
    if request.method == "OPTIONS":
        response = JSONResponse(content={}, status_code=200)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "false"
        response.headers["Access-Control-Max-Age"] = "3600"
        return response

    # Handle actual request
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "false"
    return response


# Include API routes
app.include_router(router, prefix="/api", tags=["projects"])


@app.get("/")
async def root():
    return {
        "message": "Project Portfolio RAG API",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)