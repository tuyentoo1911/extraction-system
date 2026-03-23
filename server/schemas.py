from pydantic import BaseModel


class ExtractRequest(BaseModel):
    text: str
    use_deep_analysis: bool = False


class EntityProperty(BaseModel):
    key: str
    value: str


class Entity(BaseModel):
    id: str
    name: str
    type: str
    properties: list[EntityProperty] = []


class Relation(BaseModel):
    source: str
    target: str
    label: str
    isPredicted: bool = False


class GraphData(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


class PredictLinksRequest(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
    use_deep_analysis: bool = False


class PredictLinksResponse(BaseModel):
    predicted_relations: list[Relation]


class MetricsRequest(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


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
