from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


NOISE_PATTERNS = [
    re.compile(r"^\d+(?:[.,]\d+)?%$"),
    re.compile(r"^\d+(?:[.,]\d+)?%/\w+$"),
    re.compile(r"^năm(?:\s+\d{4})?$", re.IGNORECASE),
    re.compile(r"^\d{4}$"),
    re.compile(r"^\d+(?:[.,]\d+)?$"),
]

GENERIC_TERMS = {
    "công nghiệp",
    "công nghệ",
    "công",
    "tập đoàn",
    "doanh nghiệp",
    "thị trường",
    "ngân hàng",
    "kinh tế",
}


@dataclass(frozen=True)
class InsightConfig:
    top_k: int
    anomaly_k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate graph/ML insights and BI-ready outputs."
    )
    parser.add_argument(
        "--node-metrics",
        default="data/graph_metrics/node_metrics.csv",
        help="Path to node metrics CSV.",
    )
    parser.add_argument(
        "--summary",
        default="data/graph_metrics/summary.json",
        help="Path to graph summary JSON.",
    )
    parser.add_argument(
        "--feature-data",
        default="feature_engineering_output/node_features.csv",
        help="Optional feature engineering output with labels/predictions.",
    )
    parser.add_argument(
        "--prediction-data",
        default="src/model/influence_predictor/test_predictions.csv",
        help="Optional ML predictions CSV.",
    )
    parser.add_argument(
        "--model-metadata",
        default="src/model/influence_predictor/metadata.json",
        help="Optional model metadata JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/insights",
        help="Directory for generated insight files.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top entities to include per section.",
    )
    parser.add_argument(
        "--anomaly-k",
        type=int,
        default=8,
        help="Number of anomaly/outlier rows to show.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export BI-oriented CSV tables. Disabled by default.",
    )
    return parser.parse_args()


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify_flow_pattern(df: pd.DataFrame) -> pd.Series:
    ratio = (df["out_degree"] + 1.0) / (df["in_degree"] + 1.0)
    labels: list[str] = []
    for value in ratio:
        if value >= 3.0:
            labels.append("broadcaster")
        elif value <= 1 / 3:
            labels.append("collector")
        else:
            labels.append("balanced")
    return pd.Series(labels, index=df.index)


def classify_influence_band(df: pd.DataFrame) -> pd.Series:
    high = df["influence_score"].quantile(0.9)
    mid = df["influence_score"].quantile(0.6)
    labels: list[str] = []
    for value in df["influence_score"]:
        if value >= high:
            labels.append("HIGH")
        elif value >= mid:
            labels.append("MEDIUM")
        else:
            labels.append("LOW")
    return pd.Series(labels, index=df.index)


def is_noise_like(node: Any, entity_type: Any) -> bool:
    text = str(node or "").strip().lower()
    entity_text = str(entity_type or "").strip().upper()
    if not text:
        return True
    if text in GENERIC_TERMS:
        return True
    if any(pattern.match(text) for pattern in NOISE_PATTERNS):
        return True
    if entity_text == "UNKNOWN" and len(text) <= 4:
        return True
    return False


def seems_mojibake(node: Any) -> bool:
    text = str(node or "")
    return any(token in text for token in ("Ã", "Ä", "Å", "á»", "â"))


def prepare_node_metrics(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["flow_pattern"] = classify_flow_pattern(prepared)
    prepared["heuristic_influence_label"] = classify_influence_band(prepared)
    prepared["degree_gap"] = prepared["out_degree"] - prepared["in_degree"]
    prepared["strength_gap"] = prepared["out_strength"] - prepared["in_strength"]
    prepared["pr_per_degree"] = prepared["pagerank"] / prepared["total_degree"].clip(lower=1)
    prepared["betweenness_x_diversity"] = (
        prepared["betweenness"] * prepared["relation_diversity"].clip(lower=1)
    )
    prepared["is_noise_like"] = prepared.apply(
        lambda row: is_noise_like(row["node"], row["entity_type"]), axis=1
    )
    prepared["has_encoding_issue"] = prepared["node"].map(seems_mojibake)
    prepared["analysis_quality"] = "usable"
    prepared.loc[prepared["is_noise_like"], "analysis_quality"] = "review"
    prepared.loc[prepared["has_encoding_issue"], "analysis_quality"] = "encoding_review"
    return prepared


def _to_jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def top_records(
    df: pd.DataFrame,
    sort_col: str,
    columns: list[str],
    n: int,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    if df.empty:
        return []
    subset = df.sort_values(sort_col, ascending=ascending).head(n)
    return [
        {column: _to_jsonable(row[column]) for column in columns}
        for _, row in subset.iterrows()
    ]


def graph_overview(node_df: pd.DataFrame, summary: dict[str, Any] | None) -> dict[str, Any]:
    graph_data = (summary or {}).get("graph", {})
    return {
        "nodes": int(graph_data.get("nodes", len(node_df))),
        "edges_unique": int(graph_data.get("edges_unique", 0)),
        "density": float(graph_data.get("density", 0.0)),
        "weakly_connected_components": int(graph_data.get("weakly_connected_components", 0)),
        "strongly_connected_components": int(
            graph_data.get("strongly_connected_components", 0)
        ),
        "relation_types": int(graph_data.get("relation_types", 0)),
        "median_degree": float(node_df["total_degree"].median()),
        "median_influence": float(node_df["influence_score"].median()),
        "high_influence_nodes": int(
            (node_df["heuristic_influence_label"] == "HIGH").sum()
        ),
        "noise_like_nodes": int(node_df["is_noise_like"].sum()),
        "encoding_issue_nodes": int(node_df["has_encoding_issue"].sum()),
    }


def build_entity_type_summary(node_df: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        node_df.groupby("entity_type", dropna=False)
        .agg(
            node_count=("node", "count"),
            median_degree=("total_degree", "median"),
            mean_influence=("influence_score", "mean"),
            high_influence_nodes=("heuristic_influence_label", lambda s: int((s == "HIGH").sum())),
            noise_like_nodes=("is_noise_like", "sum"),
        )
        .reset_index()
        .sort_values(["mean_influence", "node_count"], ascending=[False, False])
    )
    return [
        {
            "entity_type": _to_jsonable(row["entity_type"]),
            "node_count": int(row["node_count"]),
            "median_degree": float(row["median_degree"]),
            "mean_influence": float(row["mean_influence"]),
            "high_influence_nodes": int(row["high_influence_nodes"]),
            "noise_like_nodes": int(row["noise_like_nodes"]),
        }
        for _, row in grouped.iterrows()
    ]


def build_quality_insights(node_df: pd.DataFrame, cfg: InsightConfig) -> dict[str, Any]:
    return {
        "noise_ratio": float(node_df["is_noise_like"].mean()),
        "encoding_issue_ratio": float(node_df["has_encoding_issue"].mean()),
        "noise_candidates": top_records(
            node_df[node_df["is_noise_like"]],
            "influence_score",
            [
                "node",
                "entity_type",
                "total_degree",
                "influence_score",
                "flow_pattern",
                "analysis_quality",
            ],
            cfg.anomaly_k,
        ),
        "encoding_candidates": top_records(
            node_df[node_df["has_encoding_issue"]],
            "influence_score",
            [
                "node",
                "entity_type",
                "total_degree",
                "influence_score",
                "analysis_quality",
            ],
            cfg.anomaly_k,
        ),
    }


def build_graph_insights(
    node_df: pd.DataFrame,
    summary: dict[str, Any] | None,
    cfg: InsightConfig,
) -> dict[str, Any]:
    clean_df = node_df[~node_df["is_noise_like"]].copy()
    columns_common = [
        "node",
        "entity_type",
        "total_degree",
        "pagerank",
        "betweenness",
        "relation_diversity",
        "influence_score",
    ]
    flow_columns = [
        "node",
        "entity_type",
        "in_degree",
        "out_degree",
        "degree_gap",
        "flow_pattern",
        "influence_score",
    ]

    graph_sections = {
        "overview": graph_overview(node_df, summary),
        "top_influence": top_records(clean_df, "influence_score", columns_common, cfg.top_k),
        "top_pagerank": top_records(clean_df, "pagerank", columns_common, cfg.top_k),
        "top_brokers": top_records(
            clean_df,
            "betweenness_x_diversity",
            columns_common + ["betweenness_x_diversity"],
            cfg.top_k,
        ),
        "top_broadcasters": top_records(
            clean_df[clean_df["flow_pattern"] == "broadcaster"],
            "degree_gap",
            flow_columns,
            cfg.top_k,
        ),
        "top_collectors": top_records(
            clean_df[clean_df["flow_pattern"] == "collector"],
            "degree_gap",
            flow_columns,
            cfg.top_k,
            ascending=True,
        ),
        "high_confidence_low_connectivity": top_records(
            clean_df[clean_df["total_degree"] <= clean_df["total_degree"].quantile(0.25)],
            "avg_edge_confidence",
            [
                "node",
                "entity_type",
                "total_degree",
                "avg_edge_confidence",
                "max_edge_confidence",
                "total_edge_frequency",
            ],
            cfg.anomaly_k,
        ),
        "top_by_entity_type": {
            entity_type: top_records(
                clean_df[clean_df["entity_type"] == entity_type],
                "influence_score",
                columns_common,
                min(5, cfg.top_k),
            )
            for entity_type in sorted(clean_df["entity_type"].dropna().unique())
        },
        "entity_type_summary": build_entity_type_summary(node_df),
        "quality": build_quality_insights(node_df, cfg),
    }
    return graph_sections


def build_ml_insights(
    node_df: pd.DataFrame,
    feature_df: pd.DataFrame | None,
    prediction_df: pd.DataFrame | None,
    model_metadata: dict[str, Any] | None,
    cfg: InsightConfig,
) -> dict[str, Any]:
    ml: dict[str, Any] = {
        "available": False,
        "summary": [],
        "class_distribution": {},
        "top_high_label_candidates": [],
        "disagreement_candidates": [],
    }

    merged = None
    if feature_df is not None and "node" in feature_df.columns:
        merged = node_df.merge(feature_df, on="node", how="left", suffixes=("", "_feature"))

    label_col = None
    if merged is not None:
        for candidate in ("influence_label", "predicted_label", "label"):
            if candidate in merged.columns:
                label_col = candidate
                break

    if label_col is not None:
        ml["available"] = True
        class_counts = merged[label_col].fillna("UNKNOWN").value_counts().to_dict()
        ml["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}
        top_high = merged[merged[label_col] == "HIGH"]
        ml["top_high_label_candidates"] = top_records(
            top_high,
            "influence_score",
            ["node", "entity_type", "influence_score", label_col, "pagerank", "betweenness"],
            cfg.top_k,
        )

        disagreement = merged[
            merged[label_col].fillna("UNKNOWN") != merged["heuristic_influence_label"]
        ].copy()
        if not disagreement.empty:
            disagreement["disagreement_gap"] = (
                disagreement["influence_score"] - disagreement["influence_score"].median()
            ).abs()
            ml["disagreement_candidates"] = top_records(
                disagreement,
                "disagreement_gap",
                [
                    "node",
                    "entity_type",
                    label_col,
                    "heuristic_influence_label",
                    "influence_score",
                    "total_degree",
                ],
                cfg.anomaly_k,
            )

    if prediction_df is not None and not prediction_df.empty:
        ml["available"] = True
        if {"y_true_label", "y_pred_label"}.issubset(prediction_df.columns):
            eval_df = prediction_df.copy()
            ml["test_label_distribution"] = {
                str(k): int(v)
                for k, v in eval_df["y_pred_label"].fillna("UNKNOWN").value_counts().to_dict().items()
            }
            if "pred_proba_max" in eval_df.columns:
                confident_errors = eval_df[eval_df["y_true_label"] != eval_df["y_pred_label"]].copy()
                confident_errors = confident_errors.sort_values("pred_proba_max", ascending=False)
                ml["confident_test_errors"] = [
                    {
                        "y_true_label": row["y_true_label"],
                        "y_pred_label": row["y_pred_label"],
                        "pred_proba_max": _to_jsonable(row.get("pred_proba_max")),
                    }
                    for _, row in confident_errors.head(cfg.anomaly_k).iterrows()
                ]

    if model_metadata:
        ml["available"] = True
        ml["model"] = {
            "selected_model": model_metadata.get("selected_model"),
            "val_metrics": model_metadata.get("val_metrics"),
            "test_metrics": model_metadata.get("test_metrics"),
        }
        summary_lines = []
        selected_model = model_metadata.get("selected_model")
        if selected_model:
            summary_lines.append(f"Selected model: {selected_model}")
        test_metrics = model_metadata.get("test_metrics") or {}
        macro_f1 = test_metrics.get("macro_f1")
        accuracy = test_metrics.get("accuracy")
        if macro_f1 is not None and accuracy is not None:
            summary_lines.append(
                f"Test macro-F1={macro_f1:.4f}, accuracy={accuracy:.4f}"
            )
        ml["summary"] = summary_lines

    return ml


def build_narrative(report: dict[str, Any]) -> list[str]:
    graph = report["graph"]
    overview = graph["overview"]
    top_influence = graph["top_influence"]
    top_brokers = graph["top_brokers"]
    quality = graph["quality"]
    entity_summary = graph["entity_type_summary"]

    lines: list[str] = []
    if overview["median_degree"] <= 1.0:
        lines.append(
            "Đồ thị rất thưa; median degree chỉ khoảng 1, nên phần lớn thực thể mới chỉ có một số kết nối rất hạn chế."
        )
    if overview["weakly_connected_components"] > overview["nodes"] * 0.2:
        lines.append(
            "Mạng đang phân mảnh mạnh với nhiều weakly connected components, vì vậy insight hiện nghiêng về centrality cục bộ hơn là cấu trúc cụm lớn."
        )
    if top_influence:
        top_names = ", ".join(str(item["node"]) for item in top_influence[:3])
        lines.append(
            f"Nhóm ảnh hưởng cao nhất hiện tại tập trung ở {top_names}; đây là các node nên ưu tiên đưa lên dashboard hoặc báo cáo điều hành."
        )
    if top_brokers:
        broker_names = ", ".join(str(item["node"]) for item in top_brokers[:3])
        lines.append(
            f"Các broker nổi bật như {broker_names} có vai trò nối nhiều kiểu quan hệ, phù hợp để xem như hub trung gian trong mạng."
        )
    noise_ratio = quality["noise_ratio"]
    if noise_ratio >= 0.05:
        lines.append(
            f"Tỷ lệ node nghi nhiễu khoảng {noise_ratio:.1%}; cần cảnh báo khi dùng các bảng top list vì node generic có thể chen vào insight."
        )
    if overview["encoding_issue_nodes"] > 0:
        lines.append(
            f"Phát hiện {overview['encoding_issue_nodes']} node có dấu hiệu lỗi encoding; nên sửa upstream trước khi trình bày chính thức."
        )
    if entity_summary:
        best_type = entity_summary[0]["entity_type"]
        lines.append(
            f"Theo trung bình influence score, nhóm entity_type nổi bật nhất hiện tại là {best_type}."
        )
    return lines


def build_bi_handoff(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_dataset": "data/insights/node_insight_table.csv",
        "supporting_datasets": [
            "data/insights/top_rankings.csv",
            "data/insights/overview_kpis.csv",
            "data/insights/entity_type_summary.csv",
        ],
        "recommended_visuals": [
            {
                "visual": "Top influence bar chart",
                "source": "top_rankings.csv",
                "filter": "ranking_group = top_influence",
            },
            {
                "visual": "Broker network spotlight",
                "source": "top_rankings.csv",
                "filter": "ranking_group = top_brokers",
            },
            {
                "visual": "Entity type distribution",
                "source": "entity_type_summary.csv",
                "filter": "none",
            },
            {
                "visual": "Quality warnings",
                "source": "overview_kpis.csv",
                "filter": "kpi_name in (noise_like_nodes, encoding_issue_nodes)",
            },
        ],
        "notes": [
            "node_insight_table.csv is the main flat table for slicing by entity_type, flow_pattern, heuristic_influence_label, and quality flags.",
            "top_rankings.csv is pre-ranked to reduce Power BI transformation effort.",
            "Rows marked analysis_quality != usable should be reviewed before being highlighted in dashboards.",
        ],
    }


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_table(lines: list[str], title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines.append(f"## {title}")
    if not rows:
        lines.append("- No data available.")
        lines.append("")
        return
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = [format_value(row.get(col)) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    graph = report["graph"]
    quality = graph["quality"]
    ml = report["ml"]
    overview = graph["overview"]

    lines.append("# Automatic Insights Report")
    lines.append("")
    lines.append("## Executive Summary")
    for item in report["narrative"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Graph Overview")
    lines.append(
        f"- Graph has {overview['nodes']:,} nodes, {overview['edges_unique']:,} unique edges, density {overview['density']:.6f}."
    )
    lines.append(
        f"- Weakly connected components: {overview['weakly_connected_components']:,}; strongly connected components: {overview['strongly_connected_components']:,}."
    )
    lines.append(
        f"- Median degree is {overview['median_degree']:.2f}; median influence score is {overview['median_influence']:.4f}."
    )
    lines.append(
        f"- Heuristic high-influence segment contains {overview['high_influence_nodes']:,} nodes."
    )
    lines.append(
        f"- Noise-like nodes: {overview['noise_like_nodes']:,}; encoding issue nodes: {overview['encoding_issue_nodes']:,}."
    )
    lines.append("")

    render_table(
        lines,
        "Top Influence Nodes",
        graph["top_influence"],
        ["node", "entity_type", "total_degree", "influence_score", "pagerank", "betweenness"],
    )
    render_table(
        lines,
        "Top Broker Nodes",
        graph["top_brokers"],
        ["node", "entity_type", "betweenness_x_diversity", "relation_diversity", "influence_score"],
    )
    render_table(
        lines,
        "Top Broadcasters",
        graph["top_broadcasters"],
        ["node", "entity_type", "out_degree", "in_degree", "degree_gap", "influence_score"],
    )
    render_table(
        lines,
        "Top Collectors",
        graph["top_collectors"],
        ["node", "entity_type", "in_degree", "out_degree", "degree_gap", "influence_score"],
    )
    render_table(
        lines,
        "Entity Type Summary",
        graph["entity_type_summary"],
        ["entity_type", "node_count", "median_degree", "mean_influence", "high_influence_nodes", "noise_like_nodes"],
    )
    render_table(
        lines,
        "High Confidence Low Connectivity",
        graph["high_confidence_low_connectivity"],
        ["node", "entity_type", "total_degree", "avg_edge_confidence", "total_edge_frequency"],
    )
    render_table(
        lines,
        "Noise Candidates",
        quality["noise_candidates"],
        ["node", "entity_type", "total_degree", "influence_score", "flow_pattern", "analysis_quality"],
    )
    render_table(
        lines,
        "Encoding Candidates",
        quality["encoding_candidates"],
        ["node", "entity_type", "total_degree", "influence_score", "analysis_quality"],
    )

    lines.append("## ML Insights")
    if not ml["available"]:
        lines.append(
            "- ML artifacts were not found. Report falls back to graph-based heuristic influence labels."
        )
        lines.append("")
    else:
        for item in ml.get("summary", []):
            lines.append(f"- {item}")
        if ml.get("class_distribution"):
            dist_text = ", ".join(f"{k}: {v}" for k, v in ml["class_distribution"].items())
            lines.append(f"- Class distribution: {dist_text}")
        if ml.get("test_label_distribution"):
            dist_text = ", ".join(f"{k}: {v}" for k, v in ml["test_label_distribution"].items())
            lines.append(f"- Test prediction distribution: {dist_text}")
        lines.append("")
        render_table(
            lines,
            "Top High-Label Candidates",
            ml.get("top_high_label_candidates", []),
            ["node", "entity_type", "influence_score", "influence_label", "pagerank", "betweenness"],
        )
        render_table(
            lines,
            "Disagreement Candidates",
            ml.get("disagreement_candidates", []),
            ["node", "entity_type", "influence_label", "heuristic_influence_label", "influence_score", "total_degree"],
        )

    lines.append("## BI Handoff")
    lines.append(f"- Primary dataset: `{report['bi_handoff']['primary_dataset']}`")
    for note in report["bi_handoff"]["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def write_csv_exports(output_dir: Path, node_df: pd.DataFrame, report: dict[str, Any]) -> list[Path]:
    exported: list[Path] = []

    node_export_cols = [
        "node",
        "entity_type",
        "in_degree",
        "out_degree",
        "total_degree",
        "in_strength",
        "out_strength",
        "total_strength",
        "betweenness",
        "pagerank",
        "authority_score",
        "hub_score",
        "avg_edge_confidence",
        "total_edge_frequency",
        "relation_diversity",
        "influence_score",
        "flow_pattern",
        "heuristic_influence_label",
        "degree_gap",
        "strength_gap",
        "pr_per_degree",
        "betweenness_x_diversity",
        "is_noise_like",
        "has_encoding_issue",
        "analysis_quality",
    ]
    node_table_path = output_dir / "node_insight_table.csv"
    node_df[node_export_cols].to_csv(node_table_path, index=False, encoding="utf-8-sig")
    exported.append(node_table_path)

    ranking_rows: list[dict[str, Any]] = []
    for group_name in (
        "top_influence",
        "top_pagerank",
        "top_brokers",
        "top_broadcasters",
        "top_collectors",
        "high_confidence_low_connectivity",
    ):
        for rank, row in enumerate(report["graph"][group_name], start=1):
            ranking_rows.append(
                {
                    "ranking_group": group_name,
                    "rank": rank,
                    **row,
                }
            )
    rankings_path = output_dir / "top_rankings.csv"
    pd.DataFrame(ranking_rows).to_csv(rankings_path, index=False, encoding="utf-8-sig")
    exported.append(rankings_path)

    overview_rows = [
        {"kpi_name": key, "kpi_value": value}
        for key, value in report["graph"]["overview"].items()
    ]
    overview_rows.extend(
        [
            {"kpi_name": "noise_ratio", "kpi_value": report["graph"]["quality"]["noise_ratio"]},
            {
                "kpi_name": "encoding_issue_ratio",
                "kpi_value": report["graph"]["quality"]["encoding_issue_ratio"],
            },
        ]
    )
    overview_path = output_dir / "overview_kpis.csv"
    pd.DataFrame(overview_rows).to_csv(overview_path, index=False, encoding="utf-8-sig")
    exported.append(overview_path)

    entity_path = output_dir / "entity_type_summary.csv"
    pd.DataFrame(report["graph"]["entity_type_summary"]).to_csv(
        entity_path, index=False, encoding="utf-8-sig"
    )
    exported.append(entity_path)

    return exported


def save_report(
    output_dir: Path,
    report: dict[str, Any],
    node_df: pd.DataFrame,
    export_csv: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "automatic_insights.json",
        "md": output_dir / "automatic_insights.md",
    }
    paths["json"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    if export_csv:
        for exported in write_csv_exports(output_dir, node_df, report):
            paths[exported.stem] = exported
    return paths


def main() -> None:
    args = parse_args()
    cfg = InsightConfig(top_k=args.top_k, anomaly_k=args.anomaly_k)

    node_metrics_path = Path(args.node_metrics)
    summary_path = Path(args.summary)
    feature_path = Path(args.feature_data)
    prediction_path = Path(args.prediction_data)
    model_metadata_path = Path(args.model_metadata)
    output_dir = Path(args.output_dir)

    node_df = prepare_node_metrics(pd.read_csv(node_metrics_path))
    summary = load_optional_json(summary_path)
    feature_df = load_optional_csv(feature_path)
    prediction_df = load_optional_csv(prediction_path)
    model_metadata = load_optional_json(model_metadata_path)

    report = {
        "inputs": {
            "node_metrics": str(node_metrics_path),
            "summary": str(summary_path) if summary_path.exists() else None,
            "feature_data": str(feature_path) if feature_path.exists() else None,
            "prediction_data": str(prediction_path) if prediction_path.exists() else None,
            "model_metadata": str(model_metadata_path) if model_metadata_path.exists() else None,
        },
        "graph": build_graph_insights(node_df, summary, cfg),
        "ml": build_ml_insights(node_df, feature_df, prediction_df, model_metadata, cfg),
    }
    report["narrative"] = build_narrative(report)
    report["bi_handoff"] = build_bi_handoff(report)

    paths = save_report(output_dir, report, node_df, args.export_csv)

    print(f"Loaded node metrics: {node_metrics_path}")
    print(f"Rows: {len(node_df):,}")
    print(f"Saved JSON report: {paths['json']}")
    print(f"Saved Markdown report: {paths['md']}")
    if args.export_csv:
        print(f"Saved node table: {paths['node_insight_table']}")
        print(f"Saved rankings table: {paths['top_rankings']}")
        print(f"Saved KPI table: {paths['overview_kpis']}")
        print(f"Saved entity summary: {paths['entity_type_summary']}")
    if not report["ml"]["available"]:
        print("ML artifacts not found; graph-based heuristic insights only.")


if __name__ == "__main__":
    main()
