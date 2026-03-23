# Ánh xạ nhãn NER → loại entity trong đồ thị.
# Model mới dùng 9 nhãn khớp trực tiếp với loại graph.
# None = bỏ qua.
TYPE_MAP: dict[str, str | None] = {
    "PERSON":       "Person",
    "ORGANIZATION": "Organization",
    "LOCATION":     "Location",
    "PRODUCT":      "Product",
    "EVENT":        "Event",
    "DATE":         "Date",
    "MONEY":        "Money",
    "PERCENT":      "Percent",
    "INDUSTRY":     "Industry",
}
