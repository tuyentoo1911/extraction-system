"""
FastAPI entry point — NER Knowledge Graph Extractor
Chạy: python server/main.py   hoặc   npm run server

Cải tiến #4: Rate limiting + validation
  - SlowAPI rate limiter: /extract giới hạn 10 req/phút mỗi IP.
  - MAX_UPLOAD_BYTES: từ chối file PDF vượt 20MB trước khi đọc vào RAM.
  - /upload-pdf trả về 413 nếu file quá lớn.
  - Custom exception handler trả JSON đúng format thay vì HTML mặc định.
"""

import logging

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from schemas import (
    ExtractRequest, GraphData,
    PredictLinksRequest, PredictLinksResponse,
    MetricsRequest, MetricsResponse,
    MAX_PDF_TEXT_LENGTH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Rate limiter ───────────────────────────────────────────────────────────────
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

# Giới hạn kích thước file PDF upload (bytes)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


# ── Routes ─────────────────────────────────────────────────────────────────────

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
    Trích xuất Knowledge Graph từ văn bản.
    Rate limit: 10 request/phút mỗi IP.
    Input: tối đa 50,000 ký tự (cấu hình trong schemas.py).
    """
    _require_model()
    return build_graph(run_ner(req.text), req.text)


@app.post("/upload-pdf")
@limiter.limit("5/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """
    Đọc file PDF, trả về văn bản đã trích xuất.
    Rate limit: 5 request/phút mỗi IP.
    Giới hạn kích thước: 20MB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, detail="Chỉ hỗ trợ file PDF (.pdf)")

    # Kiểm tra kích thước trước khi đọc toàn bộ vào RAM
    # Content-Length header có thể vắng mặt nên đọc từng chunk
    chunks: list[bytes] = []
    total_bytes = 0
    async for chunk in file:
        total_bytes += len(chunk)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                detail=(
                    f"File vượt quá kích thước tối đa "
                    f"({MAX_UPLOAD_BYTES // (1024*1024)} MB). "
                    "Vui lòng upload file nhỏ hơn."
                ),
            )
        chunks.append(chunk)

    data = b"".join(chunks)

    try:
        from pypdf import PdfReader
        import io

        reader = PdfReader(io.BytesIO(data))
        pages  = [p.extract_text().strip() for p in reader.pages if p.extract_text()]

        if not pages:
            raise HTTPException(422, detail="Không trích xuất được văn bản. File có thể là PDF scan.")

        full_text = "\n\n".join(pages)

        # Cắt bớt nếu vượt giới hạn text trích xuất
        if len(full_text) > MAX_PDF_TEXT_LENGTH:
            logger.warning(
                "PDF '%s' extracted text truncated: %d → %d chars",
                file.filename, len(full_text), MAX_PDF_TEXT_LENGTH,
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
        raise HTTPException(500, detail=f"Lỗi đọc PDF: {e}")


@app.post("/predict-links", response_model=PredictLinksResponse)
@limiter.limit("30/minute")
def predict_links(request: Request, req: PredictLinksRequest):
    _require_model()
    return PredictLinksResponse(predicted_relations=predict_new_links(req.entities, req.relations))


@app.post("/metrics", response_model=MetricsResponse)
@limiter.limit("20/minute")
def metrics(request: Request, req: MetricsRequest):
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
@limiter.limit("30/minute")
def kb_search(request: Request, q: str, limit: int = 20):
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_model():
    if not model_state.model_ready:
        raise HTTPException(
            503,
            detail=f"Model chưa sẵn sàng. {model_state.model_error or 'Đang tải...'}",
        )


# ── Startup ────────────────────────────────────────────────────────────────────

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
