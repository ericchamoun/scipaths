from typing import Dict, List


def build_contribution_prompt(
    target_info: Dict[str, str],
    citing_info: Dict[str, str],
    contexts: List[str],
) -> str:
    header = [
        "You are extracting how a citing paper uses or extends a target paper.",
        "Read the paragraph(s) below and write ONE concise contribution claim.",
        "Focus only on what the citing paper actually does with the target paper.",
        "",
        "Rules:",
        "- If the citing paper explicitly uses/adopts/evaluates on the target paper's method/data/benchmark, label USES.",
        "- If it explicitly extends/modifies/adapts/builds upon the target paper, label EXTENDS.",
        "- If the paragraph is only descriptive/background or only compares/mentions the target paper, return label NOT_CONFIRMED and empty fields.",
        "- Do not output comparison-only claims (e.g., 'compares to <CITED HERE>'); those are NOT_CONFIRMED.",
        "- Output paper_claim: one concise, paper-specific contribution claim.",
        "- Output cluster_title: concise natural-language cluster summary (6-14 words), generic across papers.",
        "- Also output cluster_key in this exact format: RELATION|artifact|purpose",
        "- cluster_key must be generic and reusable across papers.",
        "- artifact and purpose must be short snake_case phrases (e.g., dataset, evaluation_protocol, evaluation).",
        "- cluster_key RELATION must exactly match label.",
        "- Avoid overly specific keys (no paper names, no model/version numbers, no citation keys).",
        "- Prefer stable generic keys such as: USES|dataset|evaluation, EXTENDS|dataset|dataset_creation, USES|evaluation_protocol|evaluation.",
        "- If label is NOT_CONFIRMED, paper_claim, cluster_title, cluster_key, and evidence_span must be empty.",
        "- The evidence_span must be a verbatim substring from the provided contexts.",
        "- The TARGET_PAPER abstract/TLDR is for background only; do not use it as evidence.",
        "",
        "Negative example (NOT_CONFIRMED):",
        "Paragraph: \"We compare our method to <CITED HERE> and other baselines.\"",
        "Output: {\"label\":\"NOT_CONFIRMED\",\"paper_claim\":\"\",\"cluster_title\":\"\",\"cluster_key\":\"\",\"evidence_span\":\"\",\"rationale\":\"Comparison only.\"}",
        "",
        "Return JSON only.",
        "",
        "TARGET_PAPER:",
        f"- title: {target_info.get('title', '')}",
        f"- first_author_last: {target_info.get('first_author_last', '')}",
        f"- year: {target_info.get('year', '')}",
        f"- tldr: {target_info.get('tldr', '')}",
        f"- abstract: {target_info.get('abstract', '')}",
        "",
        "CITING_PAPER:",
        f"- title: {citing_info.get('title', '')}",
        f"- paper_id: {citing_info.get('paper_id', '')}",
        "",
        "CONTEXTS (verbatim, same order as extracted):",
    ]
    for i, text in enumerate(contexts, start=1):
        header.append(f"({i}) {text}")

    header.append("")
    header.append("JSON OUTPUT:")
    header.append(
        "{"
        "\"label\":\"USES\","
        "\"paper_claim\":\"...\","
        "\"cluster_title\":\"Uses target dataset for evaluation\","
        "\"cluster_key\":\"USES|dataset|evaluation\","
        "\"evidence_span\":\"...\","
        "\"rationale\":\"...\""
        "}"
    )
    return "\n".join(header)
