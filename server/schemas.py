"""
Pydantic schemas — request/response models cho FastAPI.

Cải tiến #4: Thêm validation cho ExtractRequest
  - max_length trên trường text để tránh crash server với input 100MB.
  - Field(...) với constraints thay vì BaseModel trống.
"""

from pydantic import BaseModel, Field, field_validator


# ── Giới hạn độ dài text đầu vào ──────────────────────────────────────────────
# PhoBERT xử lý ~400 từ/giây; 50_000 ký tự ≈ 8_000 từ ≈ ~20 giây → hợp lý.
# Thay đổi hằng số này nếu server có tài nguyên cao hơn.
MAX_TEXT_LENGTH   = 50_000  # ký tự
MAX_PDF_TEXT_LENGTH = 200_000  # PDF nhiều trang cho phép dài hơn


class ExtractRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description=(
            f"Văn bản cần trích xuất entity. "
            f"Tối đa {MAX_TEXT_LENGTH:,} ký tự."
        ),
    )
    use_deep_analysis: bool = False

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text không được chỉ chứa khoảng trắng.")
        return v


class EntityProperty(BaseModel):
    key: str
    value: str


class Entity(BaseModel):
    id: str
    name: str
    type: str
    properties: list[EntityProperty] = []
    aliases: list[str] = []


class Relation(BaseModel):
    source: str
    target: str
    label: str
    isPredicted: bool = False


class GraphData(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


class PredictLinksRequest(BaseModel):
    entities: list[Entity] = Field(..., max_length=500)
    relations: list[Relation] = Field(..., max_length=5000)
    use_deep_analysis: bool = False


class PredictLinksResponse(BaseModel):
    predicted_relations: list[Relation]


class MetricsRequest(BaseModel):
    entities: list[Entity] = Field(..., max_length=500)
    relations: list[Relation] = Field(..., max_length=5000)


class NodeMetrics(BaseModel):
    id: str
    name: str
    type: str
    degree: int
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    pagerank: float


class GlobalMetrics(BaseModel):
    node_count: int
    edge_count: int
    density: float
    avg_degree: float
    connected_components: int


class MetricsResponse(BaseModel):
    global_metrics: GlobalMetrics
    node_metrics: list[NodeMetrics]
    top_degree: list[NodeMetrics]
    top_pagerank: list[NodeMetrics]
    top_betweenness: list[NodeMetrics]
