"""
FastAPI entry point - NER Knowledge Graph Extractor
Run: python server/main.py or npm run server

Improvements:
  - SlowAPI rate limiter: /extract limited to 10 req/min per IP.
  - MAX_UPLOAD_BYTES: reject PDF larger than 20MB before loading into RAM.
  - /upload-pdf returns 413 if file is too large.
  - Custom exception handler returns JSON instead of default HTML.
"""

import sys
import logging
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import model as model_state
import knowledge_base as kb
from model import load_model
from knowledge_base import load_kb
from ner import run_ner
from graph import build_graph, predict_new_links
from metrics import compute_graph_metrics
from insights import compute_insight_report
from schemas import (
    ExtractRequest,
    GraphData,
    PredictLinksRequest,
    PredictLinksResponse,
    MetricsRequest,
    MetricsResponse,
    InsightRequest,
    InsightResponse,
    ChatRequest,
    ChatResponse,
    SaveWorkspaceRequest,
    SaveWorkspaceResponse,
    WorkspaceSessionSummary,
    WorkspaceSessionDetail,
    MAX_PDF_TEXT_LENGTH,
)
from chat_service import handle_chat, init_chat_db
import workspace_memory as workspace_mem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(title="KGE NER API", version="1.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

@app.get("/health")
def health():
    return {
        "status": "ok" if model_state.model_ready else "loading",
        "model_ready": model_state.model_ready,
        "model_error": model_state.model_error,
        "kb_ready": kb.kb_ready,
        "kb_triples": len(kb.triples),
    }

@app.post("/extract", response_model=GraphData)
@limiter.limit("10/minute")
def extract(request: Request, req: ExtractRequest):
    """
    Extract a knowledge graph from text.
    Rate limit: 10 requests/minute per IP.
    Input: up to 50,000 characters.
    """
    _require_model()
    return build_graph(run_ner(req.text), req.text)

@app.post("/upload-pdf")
@limiter.limit("5/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """
    Read a PDF file and return extracted text.
    Rate limit: 5 requests/minute per IP.
    File size limit: 20MB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Only PDF files are supported (.pdf)")

    chunks: list[bytes] = []
    total_bytes = 0
    async for chunk in file:
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                detail=(
                    f"File exceeds maximum size ({MAX_UPLOAD_BYTES // (1024 * 1024)} MB). "
                    "Please upload a smaller file."
                ),
            )
        chunks.append(chunk)

    data = b"".join(chunks)

    try:
        from pypdf import PdfReader
        import io

        reader = PdfReader(io.BytesIO(data))
        pages = [p.extract_text().strip() for p in reader.pages if p.extract_text()]

        if not pages:
            raise HTTPException(422, detail="Could not extract text. The file may be a scanned PDF.")

        full_text = "\n\n".join(pages)
        if len(full_text) > MAX_PDF_TEXT_LENGTH:
            logger.warning(
                "PDF '%s' extracted text truncated: %d -> %d chars",
                file.filename,
                len(full_text),
                MAX_PDF_TEXT_LENGTH,
            )
            full_text = full_text[:MAX_PDF_TEXT_LENGTH]

        return {
            "text": full_text,
            "page_count": len(reader.pages),
            "extracted_pages": len(pages),
            "filename": file.filename,
            "truncated": len("\n\n".join(pages)) > MAX_PDF_TEXT_LENGTH,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"PDF read error: {e}")

@app.post("/predict-links", response_model=PredictLinksResponse)
@limiter.limit("30/minute")
def predict_links(request: Request, req: PredictLinksRequest):
    _require_model()
    return PredictLinksResponse(predicted_relations=predict_new_links(req.entities, req.relations))

@app.post("/metrics", response_model=MetricsResponse)
@limiter.limit("20/minute")
def metrics(request: Request, req: MetricsRequest):
    """Compute graph metrics from current entities/relations."""
    try:
        return compute_graph_metrics(GraphData(entities=req.entities, relations=req.relations))
    except RuntimeError as e:
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Graph metrics error: {e}")

@app.post("/insight", response_model=InsightResponse)
async def insight(request: Request, req: InsightRequest):
    """Generate markdown insight from the current graph using backend analysis."""
    try:
        return await compute_insight_report(
            GraphData(entities=req.entities, relations=req.relations),
            req.input_text,
        )
    except RuntimeError as e:
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Insight generation error: {e}")

@app.get("/kb/stats")
def kb_stats():
    """Knowledge base statistics."""
    return kb.get_stats()

@app.get("/kb/search")
@limiter.limit("30/minute")
def kb_search(request: Request, q: str, limit: int = 20):
    """Search entities in the knowledge base."""
    if not kb.kb_ready:
        raise HTTPException(503, detail="Knowledge base is not ready.")
    if not q.strip():
        raise HTTPException(400, detail="Query must not be empty.")
    return {"query": q, "results": kb.search_entities(q, limit=limit)}

@app.get("/kb/entity")
def kb_entity(name: str, limit: int = 50):
    """Get triples related to an entity."""
    if not kb.kb_ready:
        raise HTTPException(503, detail="Knowledge base is not ready.")
    triples = kb.get_entity_triples(name, limit=limit)
    return {"entity": name, "total": len(triples), "triples": triples}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, req: ChatRequest):
    """Hybrid chatbot with PostgreSQL memory and optional LLM."""
    try:
        return await handle_chat(req)
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(500, detail=f"Chat error: {e}")


@app.get("/workspace/sessions", response_model=list[WorkspaceSessionSummary])
def list_workspace_sessions(limit: int = 50):
    try:
        rows = workspace_mem.list_workspaces(limit=limit)
        return [
            WorkspaceSessionSummary(
                id=row["id"],
                title=row["title"],
                preview_text=row.get("preview_text") or "",
                entities_count=int(row.get("entities_count") or 0),
                relations_count=int(row.get("relations_count") or 0),
                created_at=row["created_at"].isoformat(),
                updated_at=row["updated_at"].isoformat(),
            )
            for row in rows
        ]
    except Exception as e:
        logger.exception("Workspace list error")
        raise HTTPException(500, detail=f"Workspace list error: {e}")


@app.get("/workspace/sessions/{session_id}", response_model=WorkspaceSessionDetail)
def get_workspace_session(session_id: str):
    try:
        row = workspace_mem.get_workspace(session_id)
        if not row:
            raise HTTPException(404, detail="Workspace session not found.")
        return WorkspaceSessionDetail(
            id=row["id"],
            title=row["title"],
            input_text=row.get("input_text") or "",
            graph_data=row.get("graph_data"),
            metrics_data=row.get("metrics_data"),
            insight_markdown=row.get("insight_markdown"),
            chat_session_id=row.get("chat_session_id"),
            chat_engine=row.get("chat_engine"),
            chat_history=row.get("chat_history") or [],
            active_tab=row.get("active_tab") or "graph",
            created_at=row["created_at"].isoformat(),
            updated_at=row["updated_at"].isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Workspace get error")
        raise HTTPException(500, detail=f"Workspace get error: {e}")


@app.post("/workspace/sessions", response_model=SaveWorkspaceResponse)
def save_workspace_session(req: SaveWorkspaceRequest):
    try:
        sid = workspace_mem.save_workspace(
            session_id=req.session_id,
            title=req.title,
            input_text=req.input_text,
            graph_data=req.graph_data.model_dump() if req.graph_data else None,
            metrics_data=req.metrics_data.model_dump() if req.metrics_data else None,
            insight_markdown=req.insight_markdown,
            chat_session_id=req.chat_session_id,
            chat_engine=req.chat_engine,
            chat_history=[turn.model_dump() for turn in req.chat_history] if req.chat_history else [],
            active_tab=req.active_tab,
        )
        return SaveWorkspaceResponse(session_id=sid)
    except Exception as e:
        logger.exception("Workspace save error")
        raise HTTPException(500, detail=f"Workspace save error: {e}")


@app.delete("/workspace/sessions/{session_id}")
def delete_workspace_session(session_id: str):
    try:
        deleted = workspace_mem.delete_workspace(session_id)
        if not deleted:
            raise HTTPException(404, detail="Workspace session not found.")
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Workspace delete error")
        raise HTTPException(500, detail=f"Workspace delete error: {e}")

def _require_model():
    if not model_state.model_ready:
        raise HTTPException(
            503,
            detail=f"Model is not ready. {model_state.model_error or 'Loading...'}",
        )

@app.on_event("startup")
async def startup_event():
    import asyncio

    loop = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(None, load_model),
        loop.run_in_executor(None, load_kb),
        loop.run_in_executor(None, init_chat_db),
        loop.run_in_executor(None, workspace_mem.init_workspace_db),
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
