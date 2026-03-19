import json
import sys

import networkx as nx
import pandas as pd


TRIPLES_PATH = "data/triples.json"
OUTPUT_PATH = "data/graph_metrics.csv"


def load_triples(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_graph(triples: list[dict]) -> nx.DiGraph:
    graph = nx.DiGraph()

    for triple in triples:
        graph.add_edge(
            triple["subject"],
            triple["object"],
            relation=triple["relation"],
        )

    return graph


def calculate_metrics(graph: nx.DiGraph) -> pd.DataFrame:
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)
    pagerank = nx.pagerank(graph)

    metrics = []
    for node in graph.nodes():
        metrics.append(
            {
                "node": node,
                "degree": degree[node],
                "betweenness": betweenness[node],
                "pagerank": pagerank[node],
            }
        )

    return pd.DataFrame(metrics)


def main() -> None:
    triples = load_triples(TRIPLES_PATH)
    graph = build_graph(triples)
    metrics_df = calculate_metrics(graph)

    metrics_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    try:
        print(metrics_df.to_string(index=False))
    except UnicodeEncodeError:
        safe_output = metrics_df.to_string(index=False).encode(
            sys.stdout.encoding or "utf-8",
            errors="replace",
        ).decode(sys.stdout.encoding or "utf-8")
        print(safe_output)


if __name__ == "__main__":
    main()
