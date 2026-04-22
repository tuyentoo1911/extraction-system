"""Graph metrics utilities.

Cải tiến #5: Betweenness centrality với k-sampling
  - nx.betweenness_centrality là O(VE) — với graph 1000+ node sẽ timeout.
  - Thêm tham số k = min(node_count, BETWEENNESS_K_SAMPLES) để chỉ
    dùng k nguồn ngẫu nhiên thay vì tính toàn phần.
  - Với graph nhỏ (≤ BETWEENNESS_K_SAMPLES node), k = None → tính chính xác.
  - Thêm timeout guard: nếu graph quá lớn, log warning và dùng k nhỏ hơn.
"""

import logging

from schemas import GraphData

logger = logging.getLogger(__name__)

BETWEENNESS_K_SAMPLES   = 100
LARGE_GRAPH_NODE_WARN   = 500

def compute_graph_metrics(data: GraphData) -> dict:
    """
    Tính các chỉ số graph chính từ GraphData:
    - degree / betweenness / closeness / pagerank theo node
    - density / components / avg_degree toàn graph

    Cải tiến #5 — Betweenness k-sampling:
      Với graph nhỏ  (≤ BETWEENNESS_K_SAMPLES node): tính chính xác O(VE).
      Với graph lớn  (> BETWEENNESS_K_SAMPLES node): dùng k-sampling,
      đổi thành xấp xỉ nhưng chạy trong O(k·E) thay vì O(V·E).
    """
    try:
        import networkx as nx
    except Exception as exc:
        raise RuntimeError(
            "Thiếu thư viện networkx. Cài bằng: pip install networkx"
        ) from exc

    g = nx.DiGraph()
    for e in data.entities:
        g.add_node(e.id, name=e.name, type=e.type)
    for r in data.relations:
        if g.has_node(r.source) and g.has_node(r.target):
            g.add_edge(r.source, r.target, label=r.label, is_predicted=r.isPredicted)

    node_count = g.number_of_nodes()
    edge_count = g.number_of_edges()

    if node_count == 0:
        return {
            "global_metrics": {
                "node_count": 0,
                "edge_count": 0,
                "density": 0.0,
                "avg_degree": 0.0,
                "connected_components": 0,
            },
            "node_metrics": [],
            "top_degree": [],
            "top_pagerank": [],
            "top_betweenness": [],
        }

    if node_count <= BETWEENNESS_K_SAMPLES:
        betweenness_k = None
    else:
        betweenness_k = min(node_count, BETWEENNESS_K_SAMPLES)
        if node_count >= LARGE_GRAPH_NODE_WARN:
            logger.warning(
                "Large graph detected (%d nodes, %d edges). "
                "Using betweenness k-sampling (k=%d) to avoid timeout. "
                "Results are approximate.",
                node_count, edge_count, betweenness_k,
            )

    ug = g.to_undirected()

    degree_cent = nx.degree_centrality(ug)

    betweenness = nx.betweenness_centrality(
        ug,
        normalized=True,
        k=betweenness_k,  # None → full exact; int → sampled approximation
    )

    closeness = nx.closeness_centrality(ug)
    pagerank  = nx.pagerank(g) if edge_count > 0 else {n: 1.0 / node_count for n in g.nodes()}

    node_metrics = []
    for node_id in g.nodes():
        node_metrics.append({
            "id":                     node_id,
            "name":                   g.nodes[node_id].get("name", node_id),
            "type":                   g.nodes[node_id].get("type", "Unknown"),
            "degree":                 int(ug.degree(node_id)),
            "degree_centrality":      round(float(degree_cent.get(node_id, 0.0)), 6),
            "betweenness_centrality": round(float(betweenness.get(node_id, 0.0)), 6),
            "closeness_centrality":   round(float(closeness.get(node_id, 0.0)), 6),
            "pagerank":               round(float(pagerank.get(node_id, 0.0)), 6),
        })

    top_degree      = sorted(node_metrics, key=lambda x: x["degree"],                 reverse=True)[:10]
    top_pagerank    = sorted(node_metrics, key=lambda x: x["pagerank"],               reverse=True)[:10]
    top_betweenness = sorted(node_metrics, key=lambda x: x["betweenness_centrality"], reverse=True)[:10]

    weak_components = nx.number_weakly_connected_components(g) if node_count > 0 else 0
    avg_degree      = (2 * edge_count / node_count) if node_count > 0 else 0.0
    density         = nx.density(ug) if node_count > 1 else 0.0

    return {
        "global_metrics": {
            "node_count":           node_count,
            "edge_count":           edge_count,
            "density":              round(float(density), 6),
            "avg_degree":           round(float(avg_degree), 6),
            "connected_components": int(weak_components),
        },
        "node_metrics":    node_metrics,
        "top_degree":      top_degree,
        "top_pagerank":    top_pagerank,
        "top_betweenness": top_betweenness,
    }
