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

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank.")
        return v

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    engine: str = Field(description="'llm' or 'rule-based'")
    history: list[ChatTurn] = Field(
        default_factory=list,
        description="Recent conversation turns (newest last).",
    )
