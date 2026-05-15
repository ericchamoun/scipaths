CONTRIBUTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["USES", "EXTENDS", "NOT_CONFIRMED"]},
        "paper_claim": {"type": "string"},
        "cluster_title": {"type": "string"},
        "cluster_key": {"type": "string"},
        "evidence_span": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "paper_claim", "cluster_title", "cluster_key", "evidence_span", "rationale"],
}
