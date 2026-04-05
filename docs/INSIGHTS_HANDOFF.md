# Insights Handoff

## Scope
- Phần này bàn giao lớp `insights`, không bao gồm Power BI.
- Mục tiêu là cung cấp dữ liệu và diễn giải để downstream dựng dashboard hoặc báo cáo.

## Output chính
- `data/insights/automatic_insights.md`: báo cáo đọc nhanh, có executive summary.
- `data/insights/automatic_insights.json`: output có cấu trúc để downstream parse.
- `data/insights/node_insight_table.csv`: bảng phẳng chính để filter/slice.
- `data/insights/top_rankings.csv`: top ranking đã được tiền xử lý.
- `data/insights/overview_kpis.csv`: KPI tổng quan của graph và chất lượng dữ liệu.
- `data/insights/entity_type_summary.csv`: thống kê theo loại thực thể.

## Ý nghĩa các nhóm insight
- `top_influence`: các node có influence score cao nhất.
- `top_brokers`: các node đóng vai trò trung gian, dùng `betweenness_x_diversity`.
- `top_broadcasters`: node có xu hướng phát quan hệ ra ngoài.
- `top_collectors`: node có xu hướng được nhiều node khác trỏ vào.
- `high_confidence_low_connectivity`: node ít kết nối nhưng edge confidence cao.
- `quality.noise_candidates`: node nghi nhiễu hoặc generic.
- `quality.encoding_candidates`: node có dấu hiệu lỗi encoding.

## Cột quan trọng trong `node_insight_table.csv`
- `flow_pattern`: `broadcaster`, `collector`, `balanced`
- `heuristic_influence_label`: nhãn suy luận từ influence score, không phải nhãn ML thật
- `is_noise_like`: cờ nghi nhiễu
- `has_encoding_issue`: cờ nghi lỗi encoding
- `analysis_quality`: `usable`, `review`, `encoding_review`

## Caveat
- Insight hiện vẫn phụ thuộc mạnh vào chất lượng KG upstream.
- Node generic như `%`, `năm`, cụm ngành chung có thể làm sai trọng tâm nếu không lọc.
- Một phần text tiếng Việt trong dữ liệu đầu vào đang có dấu hiệu mojibake; cần sửa upstream nếu dùng cho trình bày chính thức.
- Nếu chưa có artifact ML thì phần ML insight sẽ không xuất hiện và report chỉ dùng heuristic từ graph metrics.

## Gợi ý cho downstream
- Ưu tiên visual từ `top_rankings.csv` và `overview_kpis.csv`.
- Không nên highlight các row có `analysis_quality != usable` nếu chưa review.
- Khi cần drill-down, dùng `node_insight_table.csv` làm bảng chính.
