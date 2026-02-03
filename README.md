# Project Portfolio RAG API - Backend

FastAPI backend for the Portfolio AI Assistant chatbot using RAG (Retrieval Augmented Generation) with LangChain.

## 🚀 Features

- **Vector Search** - ChromaDB for semantic similarity search
- **LangChain Integration** - Composable RAG chains and retrievers
- **AI-Powered Responses** - OpenAI GPT integration
- **RESTful API** - FastAPI with automatic docs
- **Project Management** - Query portfolio projects via natural language

## 📋 Prerequisites

- Python 3.11+
- OpenAI API key

## 🛠️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/rizovamagdalena/portfolio-ai-backend.git
cd portfolio-ai-backend
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create `.env` file:
```env
OPENAI_API_KEY=your-api-key-here
```

### 5. Ingest Data
```bash
python scripts/ingest.py
```

### 6. Run the Server
```bash
python -m uvicorn app.main:app --reload
```

API will be available at `http://localhost:8000`

## 📚 API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 Endpoints

### `GET /api/health`
Check API health and database status

### `POST /api/query`
Query projects with natural language

**Request:**
```json
{
  "query": "Which projects use AI?",
  "top_k": 3
}
```

**Response:**
```json
{
  "answer": "Based on the context...",
  "sources": [...],
  "projects_searched": [...]
}
```

### `GET /api/projects`
List all available projects

## 📁 Project Structure
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app & CORS
│   ├── api.py           # API endpoints
│   ├── schemas.py       # Pydantic models
│   ├── vector_store.py  # ChromaDB integration
│   └── llm.py           # OpenAI integration
├── data/
│   ├── projects.json    # Project data
│   └── chroma_db/       # Vector database (not in git)
├── scripts/
│   ├── ingest.py        # Data ingestion script
│   └── query.py         # Test query script
├── .env                 # Environment variables (not in git)
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔧 Technologies

- **FastAPI** - Modern Python web framework
- **ChromaDB** - Vector database
- **OpenAI API** - Embeddings and LLM
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

## 🌐 Frontend

Frontend repo: [portfolio-ai-assistant](https://github.com/rizovamagdalena/portfolio-ai-assistant)

## 👤 Author

Magdalena Rizova - [@rizovamagdalena](https://github.com/rizovamagdalena)