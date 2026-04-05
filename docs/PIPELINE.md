## Pipeline end-to-end

Tài liệu này mô tả luồng xử lý tối thiểu để tạo Knowledge Graph từ văn bản và xuất thống kê đồ thị theo hiện trạng repo.

## 1) Tạo triples (Knowledge Graph edges)
Script: `src/build_triples.py`

Hiện tại:
- Dùng model NER HuggingFace `NlpHUST/ner-vietnamese-electra-base`
- Trích xuất entity từ 1 câu text demo
- Tạo quan hệ theo luật (rule-based) dựa trên keyword trong câu
- Lưu triples ra `data/triples.json`

Chạy:
```bash
python src/build_triples.py
```

Đầu ra:
- `data/triples.json`

## 2) Tính metrics đồ thị
Script: `src/graph_metrics.py`

Chạy:
```bash
python src/graph_metrics.py
```

Đầu ra:
- `data/graph_metrics.csv`

## 3) Notebook liên quan
- `notebooks/labeling.ipynb`: phục vụ quy trình labeling/chuẩn bị dữ liệu NER
- `notebooks/knowledge_graph.ipynb`: thử nghiệm/khai phá KG (tuỳ nội dung notebook)

## 4) Gợi ý mở rộng (khi cần)
- Thay rule-based relation bằng mô hình Relation Extraction thật
- Mở rộng `src/build_triples.py` để đọc input từ file (CSV/JSON) thay vì text demo
- Chuẩn hoá schema triples (ID, type, provenance, confidence, ...)
