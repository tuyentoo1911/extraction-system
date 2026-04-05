## Dataset & định dạng dữ liệu

## Nguồn dataset
Dataset gốc được lưu trên Google Drive:
- `https://drive.google.com/drive/folders/1oM9UEoPlwUbfX1oNMy4SvRJydD-jNzuu`

## Dữ liệu trong repo

### `CSV/`
Chứa các file gán nhãn cho bài toán NER, ví dụ:
- `ner_labeled_data.csv`, `ner_labeled_data2.csv`, ...
- `ner_full_labeled.csv`

### `JSON/`
Chứa dữ liệu NER dưới dạng JSON, ví dụ:
- `ner_labeled.json`, `ner_labeled2.json`, ...
- `ner_content_labeled.json`, `ner_title_labeled.json`

## Output dữ liệu (tạo trong quá trình chạy)

### `data/triples.json`
Được tạo bởi `src/build_triples.py`.

Định dạng:
- **subject**: string
- **relation**: string
- **object**: string

Ví dụ:
```json
[
  { "subject": "Vingroup", "relation": "invest_in", "object": "VinFast" }
]
```

### `data/graph_metrics.csv`
Được tạo bởi `src/graph_metrics.py`.

Các cột:
- **node**
- **degree**
- **betweenness**
- **pagerank**
