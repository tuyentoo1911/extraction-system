# Automatic Insights Report

## Graph Overview
- Graph has 7,288 nodes, 6,226 unique edges, density 0.000117.
- Weakly connected components: 1,895; strongly connected components: 6,784.
- Median degree is 1.00; median influence score is 0.0026.
- Heuristic high-influence segment contains 729 nodes.
- Noise-like nodes: 663; encoding issue nodes: 516.

## Top Influence Nodes
| node | entity_type | total_degree | influence_score | pagerank | betweenness |
| --- | --- | --- | --- | --- | --- |
| FPT | ORGANIZATION | 91 | 0.719785 | 0.003111 | 0.001645 |
| Viettel | ORGANIZATION | 69 | 0.639422 | 0.002605 | 0.001863 |
| Apple | ORGANIZATION | 73 | 0.446090 | 0.001029 | 0.001587 |
| Bộ Công an | ORGANIZATION | 28 | 0.410056 | 0.001749 | 0.001427 |
| Google | ORGANIZATION | 40 | 0.385665 | 0.001389 | 0.001401 |
| UBND | ORGANIZATION | 42 | 0.355784 | 0.001983 | 0.000408 |
| Việt Nam | LOCATION | 107 | 0.355463 | 0.000663 | 0.000772 |
| Fed | ORGANIZATION | 16 | 0.315095 | 0.000774 | 0.000007 |
| Bộ Tài chính | ORGANIZATION | 24 | 0.292109 | 0.001424 | 0.000771 |
| Nvidia | ORGANIZATION | 38 | 0.281589 | 0.000583 | 0.001266 |

## Top Broker Nodes
| node | entity_type | betweenness_x_diversity | relation_diversity | influence_score |
| --- | --- | --- | --- | --- |
| Apple | ORGANIZATION | 0.014285 | 9.000000 | 0.446090 |
| FPT | ORGANIZATION | 0.013163 | 8.000000 | 0.719785 |
| Viettel | ORGANIZATION | 0.013044 | 7.000000 | 0.639422 |
| Nvidia | ORGANIZATION | 0.011395 | 9.000000 | 0.281589 |
| Google | ORGANIZATION | 0.009806 | 7.000000 | 0.385665 |
| Samsung | ORGANIZATION | 0.009136 | 7.000000 | 0.245182 |
| CMC | ORGANIZATION | 0.006929 | 6.000000 | 0.266042 |
| VPBank | ORGANIZATION | 0.006257 | 7.000000 | 0.261445 |
| Bộ Công an | ORGANIZATION | 0.004282 | 3.000000 | 0.410056 |
| Việt Nam | LOCATION | 0.003862 | 5.000000 | 0.355463 |

## Top Broadcasters
| node | entity_type | out_degree | in_degree | degree_gap | influence_score |
| --- | --- | --- | --- | --- | --- |
| Việt Nam | LOCATION | 93 | 14 | 79 | 0.355463 |
| Vietlott | ORGANIZATION | 52 | 0 | 52 | 0.096226 |
| Trung Quốc | LOCATION | 46 | 5 | 41 | 0.155744 |
| Nguyễn Văn | PERSON | 27 | 0 | 27 | 0.049057 |
| Huawei | ORGANIZATION | 24 | 6 | 18 | 0.113051 |
| Nhật Bản | LOCATION | 17 | 2 | 15 | 0.069378 |
| Vũ Ngọc | PERSON | 15 | 0 | 15 | 0.026415 |
| TPHCM | LOCATION | 15 | 1 | 14 | 0.036115 |
| Nguyễn Thị | PERSON | 14 | 0 | 14 | 0.024528 |
| Comex New York | UNKNOWN | 14 | 0 | 14 | 0.024528 |

## Top Collectors
| node | entity_type | in_degree | out_degree | degree_gap | influence_score |
| --- | --- | --- | --- | --- | --- |
| Fed | ORGANIZATION | 14 | 2 | -12 | 0.315095 |
| xây dựng | INDUSTRY | 11 | 0 | -11 | 0.105649 |
| Hội nghị tổng kết | EVENT | 9 | 0 | -9 | 0.120720 |
| năng lượng | INDUSTRY | 9 | 0 | -9 | 0.095894 |
| 3.000 USD | UNKNOWN | 9 | 0 | -9 | 0.078471 |
| tài chính | INDUSTRY | 9 | 0 | -9 | 0.065323 |
| đạt 100% | PERCENT | 8 | 0 | -8 | 0.118877 |
| 1 tỷ USD | MONEY | 8 | 0 | -8 | 0.074382 |
| bảo hiểm | INDUSTRY | 9 | 2 | -7 | 0.103417 |
| 5.000 USD | UNKNOWN | 7 | 0 | -7 | 0.065800 |

## Entity Type Summary
| entity_type | node_count | median_degree | mean_influence | high_influence_nodes | noise_like_nodes |
| --- | --- | --- | --- | --- | --- |
| INDUSTRY | 56 | 3.000000 | 0.046488 | 28 | 7 |
| ORGANIZATION | 2411 | 1.000000 | 0.018680 | 533 | 2 |
| LOCATION | 133 | 1.000000 | 0.011486 | 11 | 0 |
| DATE | 30 | 2.000000 | 0.008797 | 3 | 9 |
| EVENT | 35 | 1.000000 | 0.008580 | 2 | 0 |
| PERCENT | 243 | 1.000000 | 0.008269 | 16 | 18 |
| PRODUCT | 38 | 1.000000 | 0.006562 | 3 | 0 |
| MONEY | 640 | 1.000000 | 0.006237 | 22 | 0 |
| UNKNOWN | 2403 | 1.000000 | 0.005246 | 106 | 627 |
| PERSON | 1299 | 1.000000 | 0.000905 | 5 | 0 |

## High Confidence Low Connectivity
| node | entity_type | total_degree | avg_edge_confidence | total_edge_frequency |
| --- | --- | --- | --- | --- |
| Ngô Bình Nguyên | PERSON | 1 | 0.850000 | 1.000000 |
| Đỗ Nguyệt Ánh | PERSON | 1 | 0.850000 | 1.000000 |
| Nguyễn Hoàng Hải | PERSON | 1 | 0.850000 | 1.000000 |
| Trương Sơn Lâm | PERSON | 1 | 0.850000 | 1.000000 |
| Trương Thu Hoà | PERSON | 1 | 0.850000 | 1.000000 |
| Nguyễn Thị Mai Thanh | PERSON | 1 | 0.850000 | 1.000000 |
| Trần Hùng Huy | PERSON | 1 | 0.850000 | 2.000000 |
| Gates | PERSON | 1 | 0.850000 | 1.000000 |

## Noise Candidates
| node | entity_type | total_degree | influence_score | flow_pattern | analysis_quality |
| --- | --- | --- | --- | --- | --- |
| công nghiệp | INDUSTRY | 57 | 0.510313 | collector | review |
| công nghệ | INDUSTRY | 33 | 0.317217 | collector | review |
| 100% | UNKNOWN | 23 | 0.226611 | collector | review |
| 10% | UNKNOWN | 20 | 0.185872 | collector | review |
| 20% | UNKNOWN | 18 | 0.178588 | collector | review |
| 8% | UNKNOWN | 20 | 0.161245 | collector | review |
| công | INDUSTRY | 14 | 0.128055 | collector | review |
| Tập đoàn | ORGANIZATION | 19 | 0.124618 | collector | review |

## Encoding Candidates
| node | entity_type | total_degree | influence_score | analysis_quality |
| --- | --- | --- | --- | --- |
| Ngân hàng Nhà nước | ORGANIZATION | 17 | 0.175769 | encoding_review |
| xây dựng | INDUSTRY | 11 | 0.105649 | encoding_review |
| Trung tâm Đổi mới sáng tạo Quốc gia | ORGANIZATION | 7 | 0.103732 | encoding_review |
| ngân hàng | INDUSTRY | 10 | 0.096062 | encoding_review |
| Liên đoàn Điền kinh Châu | ORGANIZATION | 4 | 0.077039 | encoding_review |
| Hiệp hội Ngân hàng Việt Nam | ORGANIZATION | 4 | 0.077039 | encoding_review |
| Hội Doanh nhân trẻ Việt Nam | ORGANIZATION | 3 | 0.075150 | encoding_review |
| Trung tâm Bảo tồn Di tích Cố đô Huế | ORGANIZATION | 5 | 0.073725 | encoding_review |

## ML Insights
- ML artifacts were not found. Report falls back to graph-based heuristic influence labels.

## BI Handoff
- Primary dataset: `data/insights/node_insight_table.csv`
- node_insight_table.csv is the main flat table for slicing by entity_type, flow_pattern, heuristic_influence_label, and quality flags.
- top_rankings.csv is pre-ranked to reduce Power BI transformation effort.
- Rows marked analysis_quality != usable should be reviewed before being highlighted in dashboards.
