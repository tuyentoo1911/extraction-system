# extraction-system

Hệ thống trích xuất thực thể (NER) và xây dựng **Knowledge Graph** từ văn bản tiếng Việt, kèm bước tính **graph metrics** để phân tích mạng lưới.

## Mục tiêu
- **NER**: gán nhãn thực thể (ví dụ: ORGANIZATION, LOCATION, ...), hỗ trợ huấn luyện/đánh giá mô hình (thư mục `phobert-ner-*`, dữ liệu gán nhãn ở `CSV/`, `JSON/`).
- **Knowledge Graph**: tạo triples (subject, relation, object) và lưu ra `data/triples.json`.
- **Graph statistics**: tính degree/betweenness/pagerank và lưu ra `data/graph_metrics.csv`.

## Dataset
Dataset gốc được lưu trên Google Drive:
- Link: `https://drive.google.com/drive/folders/1oM9UEoPlwUbfX1oNMy4SvRJydD-jNzuu`

Trong repo hiện có các file gán nhãn tại:
- `CSV/` (các file `ner_labeled_data*.csv`, ...)
- `JSON/` (các file `ner_labeled*.json`, ...)

Chi tiết: xem `docs/DATASET.md`.

## Cài đặt
Yêu cầu: Python 3.10+ (khuyến nghị dùng venv)

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Chạy nhanh (demo)
### 1) Tạo triples (NER + rule-based relations demo)
```bash
python src/build_triples.py
```
Kết quả: `data/triples.json`

### 2) Tính graph metrics
```bash
python src/graph_metrics.py
```
Kết quả: `data/graph_metrics.csv`

## Cấu trúc thư mục
- `src/`: script chạy chính (tạo triples, tính metrics)
- `data/`: output trung gian/đầu ra (triples, metrics, ...)
- `CSV/`, `JSON/`: dữ liệu NER đã/đang gán nhãn
- `notebooks/`: notebook phục vụ labeling/knowledge graph
- `phobert-ner-output/`, `phobert-ner-final/`: checkpoint và model đã train

## Tài liệu
- `docs/SETUP.md`: cài đặt & lưu ý môi trường
- `docs/DATASET.md`: mô tả dữ liệu, format, quy ước nhãn
- `docs/PIPELINE.md`: luồng xử lý end-to-end, cách tái tạo đầu ra
