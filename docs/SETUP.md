## Môi trường chạy

## Yêu cầu
- **Python**: 3.10+ (khuyến nghị 3.10/3.11)
- **OS**: Windows/macOS/Linux đều được

## Cài đặt nhanh

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Lưu ý khi chạy trên Windows PowerShell
- Nếu gặp lỗi không cho phép activate script, chạy:

```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Tệp quan trọng
- `requirements.txt`: danh sách thư viện đã pin version
- `src/build_triples.py`: demo NER + sinh triples (rule-based relation)
- `src/graph_metrics.py`: tính metric của đồ thị từ `data/triples.json`
