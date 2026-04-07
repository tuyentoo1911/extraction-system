"""
FastAPI entry point - NER Knowledge Graph Extractor
Run: python server/main.py or npm run server

Improvements:
  - SlowAPI rate limiter: /extract limited to 10 req/min per IP.
  - MAX_UPLOAD_BYTES: reject PDF larger than 20MB before loading into RAM.
  - /upload-pdf returns 413 if file is too large.
  - Custom exception handler returns JSON instead of default HTML.
"""

import logging

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
    MAX_PDF_TEXT_LENGTH,
)

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
@limiter.limit("20/minute")
def insight(request: Request, req: InsightRequest):
    """Generate markdown insight from the current graph using backend analysis."""
    try:
        return compute_insight_report(
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
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
