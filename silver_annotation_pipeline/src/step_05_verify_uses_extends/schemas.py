USES_EXTENDS_VERIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {
                        "type": "string",
                        "enum": ["USES", "EXTENDS", "NOT_CONFIRMED"],
                    },
                    "cue_span": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "label", "cue_span", "rationale"],
            },
        },
    },
    "required": ["labels"],
}
