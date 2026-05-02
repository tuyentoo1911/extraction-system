"""
Pydantic schemas - request/response models for FastAPI.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_LENGTH = 50_000
MAX_PDF_TEXT_LENGTH = 200_000

class ExtractRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description=f"Input text for entity extraction. Max {MAX_TEXT_LENGTH:,} characters.",
    )
    use_deep_analysis: bool = False

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank.")
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

class InsightRequest(BaseModel):
    entities: list[Entity] = Field(..., max_length=500)
    relations: list[Relation] = Field(..., max_length=5000)
    input_text: str = Field(default="", max_length=MAX_TEXT_LENGTH)

class InsightResponse(BaseModel):
    insight_markdown: str
    report: dict[str, Any]

MAX_CHAT_MESSAGE_LENGTH = 4_000
MAX_CHAT_HISTORY_TURNS = 50

class ChatTurn(BaseModel):
    role: str = Field(..., pattern=r"^(user|model)$")
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Existing session ID. Omit or null to start a new session.",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_CHAT_MESSAGE_LENGTH,
        description="Current user message.",
    )
    entities: list[Entity] = Field(default_factory=list, max_length=500)
    relations: list[Relation] = Field(default_factory=list, max_length=5000)
    input_text: str = Field(default="", max_length=MAX_TEXT_LENGTH)
    insight_markdown: str = Field(
        default="",
        max_length=200_000,
        description="Optional Insight tab markdown report (same workspace session).",
    )
    metrics_summary: str = Field(
        default="",
        max_length=80_000,
        description="Optional compact metrics summary text from Metrics tab.",
    )

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank.")
        return v

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    engine: str = Field(description="'ollama' | 'local' | 'rule-based'")
    history: list[ChatTurn] = Field(
        default_factory=list,
        description="Recent conversation turns (newest last).",
    )
    # ── New fields for quality observability ─────────────────────────────────
    confidence: float = Field(
        default=0.0,
        description="Heuristic confidence score 0-1 based on answer grounding.",
    )
    evidence_count: int = Field(
        default=0,
        description="Number of context items that support the answer.",
    )
    intent: str = Field(
        default="",
        description="Parsed query intent (relationship/count/summary/…).",
    )


class WorkspaceSessionSummary(BaseModel):
    id: str
    title: str
    preview_text: str = ""
    entities_count: int = 0
    relations_count: int = 0
    created_at: str
    updated_at: str


class SaveWorkspaceRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=64)
    title: Optional[str] = Field(default=None, max_length=120)
    input_text: str = Field(default="", max_length=MAX_PDF_TEXT_LENGTH)
    graph_data: Optional[GraphData] = None
    metrics_data: Optional[MetricsResponse] = None
    insight_markdown: Optional[str] = Field(default=None, max_length=200_000)
    chat_session_id: Optional[str] = Field(default=None, max_length=64)
    chat_engine: Optional[str] = Field(default=None, max_length=32)
    chat_history: list[ChatTurn] = Field(default_factory=list, max_length=MAX_CHAT_HISTORY_TURNS)
    active_tab: str = Field(default="graph", max_length=24)


class SaveWorkspaceResponse(BaseModel):
    session_id: str


class WorkspaceSessionDetail(BaseModel):
    id: str
    title: str
    input_text: str
    graph_data: Optional[GraphData] = None
    metrics_data: Optional[MetricsResponse] = None
    insight_markdown: Optional[str] = None
    chat_session_id: Optional[str] = None
    chat_engine: Optional[str] = None
    chat_history: list[ChatTurn] = Field(default_factory=list)
    active_tab: str = "graph"
    created_at: str
    updated_at: str
