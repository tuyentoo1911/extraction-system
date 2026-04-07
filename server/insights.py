from __future__ import annotations

from typing import Any

from schemas import GraphData

GENERIC_TERMS = {
    "công nghiệp",
    "công nghệ",
    "công",
    "tập đoàn",
    "doanh nghiệp",
    "thị trường",
    "ngân hàng",
    "kinh tế",
}


def _build_graph(data: GraphData):
    try:
        import networkx as nx
    except Exception as exc:
        raise RuntimeError("Thiếu thư viện networkx. Cài bằng: pip install networkx") from exc

    graph = nx.DiGraph()
    for entity in data.entities:
        graph.add_node(entity.id, name=entity.name, entity_type=entity.type)
    for relation in data.relations:
        if graph.has_node(relation.source) and graph.has_node(relation.target):
            graph.add_edge(
                relation.source,
                relation.target,
                label=relation.label,
                is_predicted=relation.isPredicted,
                weight=1.0,
            )
    return nx, graph


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if max_value <= min_value:
        return {key: 0.0 for key in values}
    return {
        key: (value - min_value) / (max_value - min_value)
        for key, value in values.items()
    }


def _flow_pattern(in_degree: int, out_degree: int) -> str:
    ratio = (out_degree + 1.0) / (in_degree + 1.0)
    if ratio >= 3.0:
        return "broadcaster"
    if ratio <= 1 / 3:
        return "collector"
    return "balanced"


def _is_noise_like(name: str, entity_type: str) -> bool:
    text = (name or "").strip().lower()
    if not text:
        return True
    if text in GENERIC_TERMS:
        return True
    if entity_type.upper() == "UNKNOWN" and len(text) <= 4:
        return True
    if text.replace(".", "", 1).isdigit():
        return True
    return False


def _top_records(rows: list[dict[str, Any]], sort_key: str, limit: int, reverse: bool = True) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row.get(sort_key, 0), reverse=reverse)[:limit]


def _format_table(title: str, rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [f"## {title}"]
    if not rows:
        lines.extend(["- No data available.", ""])
        return lines
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def compute_insight_report(data: GraphData, input_text: str = "") -> dict[str, Any]:
    nx, graph = _build_graph(data)
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    predicted_relations = sum(1 for _, _, edge in graph.edges(data=True) if edge.get("is_predicted"))

    if node_count == 0:
        report = {
            "overview": {
                "nodes": 0,
                "edges": 0,
                "predicted_relations": 0,
                "density": 0.0,
                "weakly_connected_components": 0,
                "avg_degree": 0.0,
                "source_text_length": len(input_text or ""),
            },
            "top_influence": [],
            "top_brokers": [],
            "top_broadcasters": [],
            "top_collectors": [],
            "entity_type_summary": [],
            "quality": {"noise_like_nodes": 0, "noise_ratio": 0.0},
            "narrative": ["Đồ thị chưa có dữ liệu để phân tích insight."],
        }
        return {"insight_markdown": "# Automatic Insights Report\n\nĐồ thị chưa có dữ liệu để phân tích insight.\n", "report": report}

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    total_degree = dict(graph.degree())
    pagerank = nx.pagerank(graph, weight="weight") if edge_count > 0 else {n: 1.0 / node_count for n in graph.nodes()}
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
    try:
        _, authority_score = nx.hits(graph, max_iter=500, normalized=True)
    except Exception:
        authority_score = {node: 0.0 for node in graph.nodes()}

    normalized = {
        "pagerank": _normalize(pagerank),
        "betweenness": _normalize(betweenness),
        "degree": _normalize({node: float(value) for node, value in total_degree.items()}),
        "authority": _normalize(authority_score),
    }

    node_rows: list[dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get("entity_type", "UNKNOWN")
        influence_score = (
            0.4 * normalized["pagerank"].get(node_id, 0.0)
            + 0.2 * normalized["betweenness"].get(node_id, 0.0)
            + 0.2 * normalized["degree"].get(node_id, 0.0)
            + 0.2 * normalized["authority"].get(node_id, 0.0)
        )
        row = {
            "node": attrs.get("name", node_id),
            "entity_type": node_type,
            "in_degree": in_degree.get(node_id, 0),
            "out_degree": out_degree.get(node_id, 0),
            "total_degree": total_degree.get(node_id, 0),
            "degree_gap": out_degree.get(node_id, 0) - in_degree.get(node_id, 0),
            "pagerank": round(float(pagerank.get(node_id, 0.0)), 6),
            "betweenness": round(float(betweenness.get(node_id, 0.0)), 6),
            "authority_score": round(float(authority_score.get(node_id, 0.0)), 6),
            "influence_score": round(float(influence_score), 6),
            "flow_pattern": _flow_pattern(in_degree.get(node_id, 0), out_degree.get(node_id, 0)),
            "is_noise_like": _is_noise_like(attrs.get("name", node_id), node_type),
        }
        node_rows.append(row)

    clean_rows = [row for row in node_rows if not row["is_noise_like"]]
    type_summary_map: dict[str, dict[str, Any]] = {}
    for row in node_rows:
        entity_type = row["entity_type"]
        bucket = type_summary_map.setdefault(
            entity_type,
            {"entity_type": entity_type, "node_count": 0, "total_influence": 0.0, "high_influence_nodes": 0, "noise_like_nodes": 0},
        )
        bucket["node_count"] += 1
        bucket["total_influence"] += row["influence_score"]
        if row["influence_score"] >= 0.6:
            bucket["high_influence_nodes"] += 1
        if row["is_noise_like"]:
            bucket["noise_like_nodes"] += 1

    entity_type_summary = []
    for bucket in type_summary_map.values():
        count = max(bucket["node_count"], 1)
        entity_type_summary.append(
            {
                "entity_type": bucket["entity_type"],
                "node_count": bucket["node_count"],
                "mean_influence": round(bucket["total_influence"] / count, 6),
                "high_influence_nodes": bucket["high_influence_nodes"],
                "noise_like_nodes": bucket["noise_like_nodes"],
            }
        )
    entity_type_summary = sorted(entity_type_summary, key=lambda item: (item["mean_influence"], item["node_count"]), reverse=True)

    top_influence = _top_records(clean_rows, "influence_score", 10)
    top_brokers = _top_records(clean_rows, "betweenness", 10)
    top_broadcasters = _top_records([row for row in clean_rows if row["flow_pattern"] == "broadcaster"], "degree_gap", 8)
    top_collectors = _top_records([row for row in clean_rows if row["flow_pattern"] == "collector"], "degree_gap", 8, reverse=False)
    noise_like_nodes = sum(1 for row in node_rows if row["is_noise_like"])

    overview = {
        "nodes": node_count,
        "edges": edge_count,
        "predicted_relations": predicted_relations,
        "density": round(float(nx.density(graph)) if node_count > 1 else 0.0, 6),
        "weakly_connected_components": int(nx.number_weakly_connected_components(graph)),
        "avg_degree": round((2 * edge_count / node_count), 6),
        "source_text_length": len(input_text or ""),
    }

    narrative: list[str] = []
    if overview["avg_degree"] <= 2:
        narrative.append("Đồ thị còn khá thưa; phần lớn thực thể mới chỉ có số kết nối hạn chế nên insight nên đọc theo cụm nhỏ.")
    if top_influence:
        top_names = ", ".join(item["node"] for item in top_influence[:3])
        narrative.append(f"Nhóm thực thể ảnh hưởng cao nhất hiện tại là {top_names}; đây là các node nên ưu tiên quan sát trên dashboard.")
    if top_brokers:
        broker_names = ", ".join(item["node"] for item in top_brokers[:3])
        narrative.append(f"Các broker nổi bật như {broker_names} đang đóng vai trò cầu nối giữa nhiều cụm quan hệ.")
    if noise_like_nodes:
        narrative.append(f"Phát hiện {noise_like_nodes} node có dấu hiệu generic hoặc nhiễu; cần thận trọng khi dùng các bảng xếp hạng tự động.")
    if entity_type_summary:
        narrative.append(f"Nhóm loại thực thể nổi bật nhất theo influence trung bình hiện là {entity_type_summary[0]['entity_type']}.")

    report = {
        "overview": overview,
        "top_influence": top_influence,
        "top_brokers": top_brokers,
        "top_broadcasters": top_broadcasters,
        "top_collectors": top_collectors,
        "entity_type_summary": entity_type_summary,
        "quality": {
            "noise_like_nodes": noise_like_nodes,
            "noise_ratio": round(noise_like_nodes / node_count, 6),
        },
        "narrative": narrative,
    }

    lines = ["# Automatic Insights Report", "", "## Executive Summary"]
    for item in narrative:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Graph Overview",
        f"- Graph has {overview['nodes']:,} nodes, {overview['edges']:,} edges, density {overview['density']:.6f}.",
        f"- Weakly connected components: {overview['weakly_connected_components']:,}; average degree {overview['avg_degree']:.2f}.",
        f"- Predicted relations currently shown in graph: {overview['predicted_relations']:,}.",
        "",
    ])
    lines.extend(_format_table("Top Influence Nodes", top_influence, ["node", "entity_type", "total_degree", "influence_score", "pagerank", "betweenness"]))
    lines.extend(_format_table("Top Broker Nodes", top_brokers, ["node", "entity_type", "total_degree", "betweenness", "influence_score"]))
    lines.extend(_format_table("Top Broadcasters", top_broadcasters, ["node", "entity_type", "out_degree", "in_degree", "degree_gap", "influence_score"]))
    lines.extend(_format_table("Top Collectors", top_collectors, ["node", "entity_type", "in_degree", "out_degree", "degree_gap", "influence_score"]))
    lines.extend(_format_table("Entity Type Summary", entity_type_summary, ["entity_type", "node_count", "mean_influence", "high_influence_nodes", "noise_like_nodes"]))

    return {"insight_markdown": "\n".join(lines), "report": report}
