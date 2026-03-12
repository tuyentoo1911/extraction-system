import json
import pandas as pd
import networkx as nx

# đọc file triples
with open("data/triples.json", "r", encoding="utf-8") as f:
    triples = json.load(f)

# tạo graph
G = nx.DiGraph()

for t in triples:
    subject = t["subject"]
    relation = t["relation"]
    obj = t["object"]

    G.add_edge(subject, obj, relation=relation)


# Degree centrality
degree = nx.degree_centrality(G)

# Betweenness centrality
betweenness = nx.betweenness_centrality(G)

# PageRank
pagerank = nx.pagerank(G)


metrics = []

for node in G.nodes():
    metrics.append({
        "node": node,
        "degree": degree[node],
        "betweenness": betweenness[node],
        "pagerank": pagerank[node]
    })

df = pd.DataFrame(metrics)


df.to_csv("data/graph_metrics.csv", index=False)

print(df)