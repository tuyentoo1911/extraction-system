#Load NER model
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import networkx as nx
import json

model_name = "NlpHUST/ner-vietnamese-electra-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)
# Extract entities từ text
text = "Vingroup đầu tư vào VinFast tại Hải Phòng và hợp tác với Samsung."

ner_result = ner(text)

entities = [(e["word"], e["entity_group"]) for e in ner_result]

print("NER result: ", ner_result)

# rule-based relation tạm do chưa có Relation Extraction thật
relations = []

orgs = [e[0] for e in entities if e[1] == "ORGANIZATION"]
locs = [e[0] for e in entities if e[1] == "LOCATION"]

if "đầu tư" in text:
    relations.append((orgs[0], "invest_in", orgs[1]))

if "hợp tác" in text:
    relations.append((orgs[0], "partner_with", orgs[-1]))

if "tại" in text:
    relations.append((orgs[1], "located_in", locs[0]))

print("Relations: ", relations)

# Chuyển thành triple
triples = []

for s,r,o in relations:
    triples.append({
        "subject": s,
        "relation": r,
        "object": o
    })

print("Triples: ", triples)

# Build Knowledge Graph

G = nx.DiGraph()

for t in triples:
    G.add_edge(t["subject"], t["object"], relation=t["relation"])

print("Knowledge Graph edges: ", G.edges(data=True))

# Lưu triples ra file JSON
with open("../data/triples.json", "w", encoding="utf-8") as f:
    json.dump(triples, f, ensure_ascii=False, indent=4)

print("Saved triples to ../data/triples.json")