from typing import Dict, List


USES_DEFINITION = (
    "USES: The CITING_PAPER explicitly uses/adopts/evaluates on/includes/relies on "
    "a dataset, benchmark, method, tool, or reported results from TARGET_PAPER "
    "as part of the CITING_PAPER's own methodology or evaluation."
)

EXTENDS_DEFINITION = (
    "EXTENDS: The CITING_PAPER explicitly extends/modifies/adapts/builds upon "
    "TARGET_PAPER's method/dataset/benchmark/tool."
)

NOTES_DEFINITION = (
    "NOT USES/EXTENDS: Merely describing what TARGET_PAPER introduces/offers/proposes "
    "or listing it among related work or benchmarks (without stating adoption). "
    "If no explicit adoption/extension cue is present, label NOT_CONFIRMED."
)


FEW_SHOT_USES = [
    "We use the same splits as <CITED HERE> .",
    "The Praat tool was used ( <CITED HERE> ) .",
    "CCGBank ( <CITED HERE> ) is used to train the model .",
    "This design idea was adopted from TANKA ( <CITED HERE>b ) .",
    "Our strategy is based on the approach presented by <CITED HERE> .",
]

FEW_SHOT_EXTENDS = [
    "The features can be easily obtained by modifying the TAT extraction algorithm described in ( <CITED HERE> ) .",
    "Our own work ( <CITED HERE> ) extends the first idea to paraphrase fragment extraction on monolingual parallel and comparable corpora .",
    "This article represents an extension of our previous work on unsupervised event coreference resolution ( Bejan et al. 2009 ; <CITED HERE> ) .",
    "This evaluation set-up is an improvement versus the one we previously reported ( <CITED HERE> ) , in which fixed partitions were used for training , development , and testing .",
    "The computational treatment of lexical rules proposed can be seen as an extension to the principled method discussed by Gotz and <CITED HERE> , 1996 , 1997b ) for encoding the main building block of HPSG grammars -- the implicative constraints -- as a logic program .",
]

FEW_SHOT_NOT_CONFIRMED = [
    "<CITED HERE> introduced factored SMT .",
    "See ( <CITED HERE> ) for a discussion .",
    "See , among others , ( <CITED HERE> ) .",
    "<CITED HERE> reported a correlation of r = .69 .",
    "See <CITED HERE> for further discussion .",
]


def build_uses_extends_verification_prompt(
    target_info: Dict[str, str],
    candidates: List[Dict[str, str]],
) -> str:
    header = [
        "You are verifying citation function for a TARGET paper inside a citing sentence.",
        "Be strict: lists of related work or benchmarks are NOT USES/EXTENDS unless there is an explicit action",
        "like \"use\", \"build on\", \"adopt\", \"extend\", \"based on\", \"trained on\", \"evaluate on\", \"implement\".",
        "",
        "Actor test (CRITICAL for USES/EXTENSION):",
        "- Only label USES or EXTENSION if the ACTION is performed by the CITING_PAPER.",
        "- The cue_span for USES/EXTENSION must include an explicit citing-paper actor phrase such as:",
        "  \"we\", \"our\", \"in this work\", \"in this paper\", \"we use\", \"we evaluate\",",
        "  \"our evaluation includes\", \"we extend\", \"we build on\", \"we adapt\".",
        "- If the context says the TARGET_PAPER (or some other paper/system) uses/extends something",
        "  (e.g., \"TARGET_PAPER uses...\", \"TARGET_PAPER extends...\"),",
        "  then it is NOT USES/EXTENSION. Label NOT_CONFIRMED.",
        "",
        "Task: Label each sentence as USES, EXTENDS, or NOT_CONFIRMED.",
        "Return JSON only with one entry per input sentence.",
        "",
        "Definitions:",
        f"- {USES_DEFINITION}",
        f"- {EXTENDS_DEFINITION}",
        f"- {NOTES_DEFINITION}",
        "",
        "Output rules:",
        "- label must be one of: USES, EXTENDS, NOT_CONFIRMED",
        "- cue_span: exact substring from the sentence that justifies USES/EXTENDS, else empty",
        "- rationale: one short sentence",
        "- If cue_span is empty => label must be NOT_CONFIRMED",
        "",
        "Few-shot examples:",
        "USES:",
    ]
    for ex in FEW_SHOT_USES:
        header.append(f"- {ex}")
    header.append("EXTENDS:")
    for ex in FEW_SHOT_EXTENDS:
        header.append(f"- {ex}")
    header.append("NOT_CONFIRMED:")
    for ex in FEW_SHOT_NOT_CONFIRMED:
        header.append(f"- {ex}")

    header.extend(
        [
            "",
            "TARGET_PAPER:",
            f"- title: {target_info.get('title', '')}",
            f"- first_author_last: {target_info.get('first_author_last', '')}",
            f"- year: {target_info.get('year', '')}",
            "",
            "CANDIDATES:",
        ]
    )

    for item in candidates:
        header.extend(
            [
                f"ID: {item['id']}",
                f"Citing paper: {item.get('citing_title', '')}",
                f"Sentence: {item.get('text', '')}",
                "",
            ]
        )

    header.append("JSON OUTPUT:")
    header.append("{\"labels\": [{\"id\": 1, \"label\": \"USES\", \"cue_span\": \"...\", \"rationale\": \"...\"}]}")
    return "\n".join(header)
