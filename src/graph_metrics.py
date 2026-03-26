from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute graph centrality metrics from triples data."
    )
    parser.add_argument(
        "--input",
        default="data/triples.csv",
        help="Path to triples JSON or CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/graph_metrics",
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top nodes to include in summary JSON.",
    )
    return parser.parse_args()


def load_triples(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict) and "triples" in data:
            data = data["triples"]
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list or contain a 'triples' list.")
        return data

    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported input format: {path.suffix}")


def as_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_graph(triples: list[dict]) -> tuple[nx.DiGraph, Counter]:
    graph = nx.DiGraph()
    relation_counts: Counter = Counter()

    for triple in triples:
        subject = (triple.get("subject") or "").strip()
        obj = (triple.get("object") or triple.get("obj") or "").strip()
        if not subject or not obj or subject == obj:
            continue

        subject_type = (triple.get("subject_type") or "UNKNOWN").strip() or "UNKNOWN"
        object_type = (
            triple.get("object_type") or triple.get("obj_type") or "UNKNOWN"
        ).strip() or "UNKNOWN"
        relation = (triple.get("relation") or "LIEN_QUAN").strip() or "LIEN_QUAN"
        weight = as_float(triple.get("weight"), as_float(triple.get("confidence"), 1.0))
        confidence = as_float(triple.get("confidence"), 1.0)
        frequency = as_int(triple.get("frequency"), 1)

        if not graph.has_node(subject):
            graph.add_node(subject, entity_type=subject_type)
        if not graph.has_node(obj):
            graph.add_node(obj, entity_type=object_type)

        relation_counts[relation] += 1

        if graph.has_edge(subject, obj):
            edge = graph[subject][obj]
            edge["weight"] += weight
            edge["frequency"] += frequency
            edge["triple_count"] += 1
            edge["confidence_max"] = max(edge["confidence_max"], confidence)
            edge["confidence_sum"] += confidence
            edge["relations"][relation] = edge["relations"].get(relation, 0) + 1
        else:
            graph.add_edge(
                subject,
                obj,
                weight=weight,
                frequency=frequency,
                triple_count=1,
                confidence_max=confidence,
                confidence_sum=confidence,
                relations={relation: 1},
            )

    return graph, relation_counts


def edge_stat_aggregates(graph: nx.DiGraph) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float | set[str]]] = {
        node: {
            "confidence_sum": 0.0,
            "confidence_max": 0.0,
            "frequency_sum": 0.0,
            "edge_count": 0.0,
            "relation_types": set(),
        }
        for node in graph.nodes()
    }

    for source, target, data in graph.edges(data=True):
        confidence_value = as_float(
            data.get("confidence_max"),
            as_float(data.get("confidence_sum"), 0.0),
        )
        frequency_value = as_float(data.get("frequency"), 0.0)
        relation_names = data.get("relations", {})

        for node in (source, target):
            node_stats = stats[node]
            node_stats["confidence_sum"] += confidence_value
            node_stats["confidence_max"] = max(node_stats["confidence_max"], confidence_value)
            node_stats["frequency_sum"] += frequency_value
            node_stats["edge_count"] += 1.0
            if isinstance(relation_names, dict):
                node_stats["relation_types"].update(relation_names.keys())

    result: dict[str, dict[str, float]] = {}
    for node, values in stats.items():
        edge_count = values["edge_count"] or 0.0
        result[node] = {
            "avg_edge_confidence": values["confidence_sum"] / edge_count if edge_count else 0.0,
            "max_edge_confidence": values["confidence_max"],
            "avg_edge_frequency": values["frequency_sum"] / edge_count if edge_count else 0.0,
            "total_edge_frequency": values["frequency_sum"],
            "relation_diversity": float(len(values["relation_types"])),
        }
    return result


def normalize_scores(values: dict[str, float]) -> dict[str, float]:
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


def compute_metrics(graph: nx.DiGraph) -> list[dict]:
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())
    total_degree = dict(graph.degree())
    in_strength = dict(graph.in_degree(weight="weight"))
    out_strength = dict(graph.out_degree(weight="weight"))
    total_strength = dict(graph.degree(weight="weight"))
    pagerank = nx.pagerank(graph, weight="weight")
    pagerank_unweighted = nx.pagerank(graph, weight=None)
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
    try:
        hub_score, authority_score = nx.hits(graph, max_iter=500, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hub_score = {node: 0.0 for node in graph.nodes()}
        authority_score = {node: 0.0 for node in graph.nodes()}
    edge_stats = edge_stat_aggregates(graph)

    influence_components = {
        "pagerank": normalize_scores(pagerank),
        "betweenness": normalize_scores(betweenness),
        "total_degree": normalize_scores(total_degree),
        "authority_score": normalize_scores(authority_score),
    }

    rows: list[dict] = []
    for node, data in graph.nodes(data=True):
        node_edge_stats = edge_stats[node]
        influence_score = (
            0.4 * influence_components["pagerank"][node]
            + 0.2 * influence_components["betweenness"][node]
            + 0.2 * influence_components["total_degree"][node]
            + 0.2 * influence_components["authority_score"][node]
        )
        rows.append(
            {
                "node": node,
                "entity_type": data.get("entity_type", "UNKNOWN"),
                "in_degree": in_degree[node],
                "out_degree": out_degree[node],
                "degree": total_degree[node],
                "total_degree": total_degree[node],
                "in_strength": in_strength[node],
                "out_strength": out_strength[node],
                "total_strength": total_strength[node],
                "betweenness": betweenness[node],
                "pagerank": pagerank[node],
                "pagerank_unweighted": pagerank_unweighted[node],
                "hub_score": hub_score[node],
                "authority_score": authority_score[node],
                "avg_edge_confidence": node_edge_stats["avg_edge_confidence"],
                "max_edge_confidence": node_edge_stats["max_edge_confidence"],
                "avg_edge_frequency": node_edge_stats["avg_edge_frequency"],
                "total_edge_frequency": node_edge_stats["total_edge_frequency"],
                "relation_diversity": node_edge_stats["relation_diversity"],
                "influence_score": influence_score,
            }
        )

    rows.sort(
        key=lambda row: (
            row["total_degree"],
            row["betweenness"],
            row["pagerank"],
            row["node"].lower(),
        ),
        reverse=True,
    )
    return rows


def top_rows(rows: list[dict], key: str, k: int) -> list[dict]:
    return sorted(rows, key=lambda row: (row[key], row["total_degree"]), reverse=True)[:k]


def write_outputs(
    output_dir: Path,
    rows: list[dict],
    graph: nx.DiGraph,
    relation_counts: Counter,
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "node_metrics.csv"
    json_path = output_dir / "summary.json"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node",
                "entity_type",
                "in_degree",
                "out_degree",
                "degree",
                "total_degree",
                "in_strength",
                "out_strength",
                "total_strength",
                "betweenness",
                "pagerank",
                "pagerank_unweighted",
                "hub_score",
                "authority_score",
                "avg_edge_confidence",
                "max_edge_confidence",
                "avg_edge_frequency",
                "total_edge_frequency",
                "relation_diversity",
                "influence_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges_unique": graph.number_of_edges(),
            "density": nx.density(graph),
            "strongly_connected_components": nx.number_strongly_connected_components(
                graph
            ),
            "weakly_connected_components": nx.number_weakly_connected_components(graph),
            "relation_types": len(relation_counts),
        },
        "top_by_degree": top_rows(rows, "degree", top_k),
        "top_by_influence": top_rows(rows, "influence_score", top_k),
        "top_by_betweenness": top_rows(rows, "betweenness", top_k),
        "top_by_pagerank": top_rows(rows, "pagerank", top_k),
    }

    with json_path.open("w", encoding="utf-8-sig") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    triples = load_triples(input_path)
    graph, relation_counts = build_graph(triples)
    rows = compute_metrics(graph)
    write_outputs(output_dir, rows, graph, relation_counts, args.top_k)

    print(f"Input: {input_path}")
    print(
        "Graph:",
        f"{graph.number_of_nodes()} nodes,",
        f"{graph.number_of_edges()} unique directed edges",
    )
    print(f"Output: {output_dir}")

    for label, key in (
        ("Degree", "degree"),
        ("Influence", "influence_score"),
        ("Betweenness", "betweenness"),
        ("PageRank", "pagerank"),
    ):
        print(f"\nTop {args.top_k} by {label}:")
        for idx, row in enumerate(top_rows(rows, key, min(args.top_k, 10)), start=1):
            print(
                f"{idx:>2}. {row['node']} | type={row['entity_type']} | "
                f"degree={row['total_degree']} | "
                f"betweenness={row['betweenness']:.6f} | "
                f"pagerank={row['pagerank']:.6f}"
            )


if __name__ == "__main__":
    main()
