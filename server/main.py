"""
FastAPI entry point — NER Knowledge Graph Extractor
Chạy: python server/main.py   hoặc   npm run server
"""

import logging

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import model as model_state
import knowledge_base as kb
from model import load_model
from knowledge_base import load_kb
from ner import run_ner
from graph import build_graph, predict_new_links
from metrics import compute_graph_metrics
from schemas import (
    ExtractRequest, GraphData,
    PredictLinksRequest, PredictLinksResponse,
    MetricsRequest, MetricsResponse,
)

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="KGE NER API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

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
def extract(req: ExtractRequest):
    _require_model()
    return build_graph(run_ner(req.text), req.text)


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Đọc file PDF, trả về văn bản đã trích xuất."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Chỉ hỗ trợ file PDF (.pdf)")
    try:
        from pypdf import PdfReader
        import io

        data   = await file.read()
        reader = PdfReader(io.BytesIO(data))
        pages  = [p.extract_text().strip() for p in reader.pages if p.extract_text()]

        if not pages:
            raise HTTPException(422, detail="Không trích xuất được văn bản. File có thể là PDF scan.")

        return {
            "text": "\n\n".join(pages),
            "page_count": len(reader.pages),
            "extracted_pages": len(pages),
            "filename": file.filename,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, detail=f"Lỗi đọc PDF: {e}")


@app.post("/predict-links", response_model=PredictLinksResponse)
def predict_links(req: PredictLinksRequest):
    _require_model()
    return PredictLinksResponse(predicted_relations=predict_new_links(req.entities, req.relations))


@app.post("/metrics", response_model=MetricsResponse)
def metrics(req: MetricsRequest):
    """Tính graph metrics từ dữ liệu entities/relations hiện tại."""
    try:
        return compute_graph_metrics(GraphData(entities=req.entities, relations=req.relations))
    except RuntimeError as e:
        raise HTTPException(500, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Lỗi tính graph metrics: {e}")


@app.get("/kb/stats")
def kb_stats():
    """Thống kê Knowledge Base."""
    return kb.get_stats()


@app.get("/kb/search")
def kb_search(q: str, limit: int = 20):
    """Tìm kiếm entity trong Knowledge Base."""
    if not kb.kb_ready:
        raise HTTPException(503, detail="Knowledge Base chưa sẵn sàng.")
    if not q.strip():
        raise HTTPException(400, detail="Query không được để trống.")
    return {"query": q, "results": kb.search_entities(q, limit=limit)}


@app.get("/kb/entity")
def kb_entity(name: str, limit: int = 50):
    """Lấy tất cả triples liên quan đến một entity."""
    if not kb.kb_ready:
        raise HTTPException(503, detail="Knowledge Base chưa sẵn sàng.")
    triples = kb.get_entity_triples(name, limit=limit)
    return {"entity": name, "total": len(triples), "triples": triples}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_model():
    if not model_state.model_ready:
        raise HTTPException(503, detail=f"Model chưa sẵn sàng. {model_state.model_error or 'Đang tải...'}")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    import asyncio
    loop = asyncio.get_event_loop()
    # Load NER model và Knowledge Base song song
    await asyncio.gather(
        loop.run_in_executor(None, load_model),
        loop.run_in_executor(None, load_kb),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
