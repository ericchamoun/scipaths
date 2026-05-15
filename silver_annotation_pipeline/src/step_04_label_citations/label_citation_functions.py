import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

DEEP_CITATION_ROOT = Path(__file__).resolve().parents[2] / "Deep-Citation"
if not DEEP_CITATION_ROOT.exists():
    raise SystemExit(f"Deep-Citation repo not found at {DEEP_CITATION_ROOT}")

sys.path.insert(0, str(DEEP_CITATION_ROOT))

from data import CollateFn, create_data_channels
from Model import MultiHeadLanguageModel
import torch
from torch.utils.data import DataLoader


PAPER_META_FILE = "paper_metadata.json"
USAGE_CONTEXTS_FILE = "usage_contexts.json"
OUT_FILE = "usage_context_labels.json"

LABEL_SET = [
    "Background",
    "Uses",
    "Extends",
    "CompareOrContrast",
    "Motivation",
    "Future",
]


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iter_paper_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for child in root.iterdir():
        if child.is_dir() and (child / PAPER_META_FILE).exists():
            out.append(child)
    return out


def flatten_contexts(usage: Dict[str, Any]) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    idx = 1
    for entry in usage.get("citing_papers", []) or []:
        if not isinstance(entry, dict):
            continue
        citing_title = entry.get("title") or "Unknown citing paper"
        citing_paper_id = entry.get("citing_paper_id") or ""
        for c in entry.get("contexts", []) or []:
            if not isinstance(c, dict):
                continue
            text = (c.get("text") or "").strip()
            if not text:
                continue
            contexts.append(
                {
                    "id": idx,
                    "text": text,
                    "citing_title": citing_title,
                    "citing_paper_id": citing_paper_id,
                }
            )
            idx += 1
    return contexts


def _resolve_model_name(lm: str) -> str:
    if lm == "scibert":
        return "allenai/scibert_scivocab_uncased"
    if lm == "bert":
        return "bert-base-uncased"
    if lm == "deberta":
        return "microsoft/deberta-v3-base"
    if lm == "deberta-large":
        return "microsoft/deberta-v3-large"
    return lm


def _infer_head_sizes(state_dict: Dict[str, Any]) -> List[int]:
    head_weights = [
        (k, v) for k, v in state_dict.items() if k.startswith("lns.") and k.endswith(".weight")
    ]
    head_weights.sort(key=lambda x: int(x[0].split(".")[1]))
    return [int(weight.shape[0]) for _, weight in head_weights]


class _ContextDataset:
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return (self.texts[idx], torch.tensor(0), torch.tensor(0))


def label_with_model(
    contexts: List[Dict[str, Any]],
    model_path: Path,
    data_dir: Path,
    class_definition: Path,
    lm: str,
    device: str,
    batch_size: int,
) -> Dict[int, Dict[str, Any]]:
    data_file = data_dir / "acl.tsv"
    train_data, _, _, label_names = create_data_channels(
        str(data_file),
        str(class_definition),
        lmbd=1.0,
    )
    modelname = _resolve_model_name(lm)
    state_dict = torch.load(model_path, map_location=device)
    head_sizes = _infer_head_sizes(state_dict)
    model = MultiHeadLanguageModel(
        modelname=modelname,
        device=device,
        readout="ch",
        num_classes=head_sizes,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    collate_fn = CollateFn(
        modelname=modelname,
        class_definitions=train_data.class_definitions,
        instance_weights=False,
    )
    dataset = _ContextDataset([ctx["text"] for ctx in contexts])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    outputs: Dict[int, Dict[str, Any]] = {}
    idx_offset = 0
    with torch.no_grad():
        for batched_text, labels, ds_indices, class_tokens, class_ds_indices in loader:
            ds_indices = ds_indices.to(device)
            class_ds_indices = class_ds_indices.to(device)
            logits = model(batched_text, ds_indices, class_tokens, class_ds_indices)[0]
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().tolist()
            pred_confidences = probs.max(dim=1).values.cpu().tolist()
            top2 = torch.topk(probs, k=2, dim=1).values.cpu()
            margins = (top2[:, 0] - top2[:, 1]).tolist()
            for i, pred in enumerate(preds):
                raw_label = label_names[pred]
                outputs[idx_offset + i + 1] = {
                    "id": idx_offset + i + 1,
                    "label": raw_label,
                    "confidence": float(pred_confidences[i]),
                    "confidence_margin": float(margins[i]),
                    "cue_span": "",
                    "rationale": "scibert_model",
                }
            idx_offset += len(preds)
    return outputs


def aggregate_citing_labels(labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_citing: Dict[str, List[Dict[str, Any]]] = {}
    for item in labels:
        citing_id = item.get("citing_paper_id") or ""
        by_citing.setdefault(citing_id, []).append(item)

    aggregated: List[Dict[str, Any]] = []
    for citing_id, items in by_citing.items():
        title = items[0].get("citing_title", "")
        labels_set = {it.get("label") for it in items}

        if "Extends" in labels_set:
            label = "Extends"
            evidence_ids = [it["id"] for it in items if it.get("label") == "Extends"]
        elif "Uses" in labels_set:
            label = "Uses"
            evidence_ids = [it["id"] for it in items if it.get("label") == "Uses"]
        elif "CompareOrContrast" in labels_set:
            label = "CompareOrContrast"
            evidence_ids = [
                it["id"] for it in items if it.get("label") == "CompareOrContrast"
            ]
        else:
            label = "Background"
            evidence_ids = []

        aggregated.append(
            {
                "citing_paper_id": citing_id,
                "citing_title": title,
                "label": label,
                "evidence_context_ids": evidence_ids,
            }
        )

    return aggregated


def aggregate_final_label(citing_labels: List[Dict[str, Any]]) -> str:
    labels_set = {item.get("label") for item in citing_labels}
    if "Extends" in labels_set:
        return "Extends"
    if "Uses" in labels_set:
        return "Uses"
    if "CompareOrContrast" in labels_set:
        return "CompareOrContrast"
    return "Background"


def score_for_paper(
    paper_dir: Path,
    batch_size: int,
    overwrite: bool,
    model_path: Path,
    model_data_dir: Path,
    model_class_def: Path,
    model_lm: str,
    device: str,
) -> str:
    usage_path = paper_dir / USAGE_CONTEXTS_FILE
    usage = load_json(usage_path)
    if not isinstance(usage, dict):
        return "missing_usage"

    contexts = flatten_contexts(usage)
    if not contexts:
        return "empty_contexts"

    out_path = paper_dir / OUT_FILE
    if out_path.exists() and not overwrite:
        return "skipped"

    labeled = label_with_model(
        contexts=contexts,
        model_path=model_path,
        data_dir=model_data_dir,
        class_definition=model_class_def,
        lm=model_lm,
        device=device,
        batch_size=batch_size,
    )

    labels_sorted = []
    for context in contexts:
        context_id = context["id"]
        item = labeled.get(context_id)
        if not item:
            item = {
                "id": context_id,
                "label": "Background",
                "confidence": 0.0,
                "cue_span": "",
                "rationale": "missing label",
            }
        item = dict(item)
        item["citing_paper_id"] = context.get("citing_paper_id", "")
        item["citing_title"] = context.get("citing_title", "")
        item["text"] = context.get("text", "")
        labels_sorted.append(item)

    citing_labels = aggregate_citing_labels(labels_sorted)
    payload = {
        "paper_id": usage.get("paper_id"),
        "num_contexts": len(contexts),
        "label_set": LABEL_SET,
        "labels": labels_sorted,
        "citing_paper_labels": citing_labels,
        "final_label": aggregate_final_label(citing_labels),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return "labeled"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label citation functions using a Deep-Citation checkpoint."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="runs/processed_papers",
        help="Root directory containing processed paper directories.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing usage_context_labels.json files.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Deep-Citation best_model.pt checkpoint.",
    )
    parser.add_argument(
        "--model-data-dir",
        type=str,
        default="Deep-Citation/Data",
        help="Deep-Citation data directory (for label order).",
    )
    parser.add_argument(
        "--model-class-def",
        type=str,
        default="Deep-Citation/Data/class_def.json",
        help="Deep-Citation class_def.json path.",
    )
    parser.add_argument(
        "--model-lm",
        type=str,
        default="scibert",
        help="Model name used for the Deep-Citation checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference (cuda/cpu).",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    paper_dirs = sorted(iter_paper_dirs(root), key=lambda p: p.name)
    print(f"[INFO] Found {len(paper_dirs)} paper dirs under {root}")

    counts = {
        "labeled": 0,
        "skipped": 0,
        "missing_usage": 0,
        "empty_contexts": 0,
    }

    for paper_dir in paper_dirs:
        status = score_for_paper(
            paper_dir,
            args.batch_size,
            args.overwrite,
            model_path=model_path,
            model_data_dir=Path(args.model_data_dir).expanduser().resolve(),
            model_class_def=Path(args.model_class_def).expanduser().resolve(),
            model_lm=args.model_lm,
            device=args.device,
        )
        counts[status] = counts.get(status, 0) + 1
        print(f"[{status.upper()}] {paper_dir.name}")

    print(
        "[SUMMARY] labeled={labeled}, skipped={skipped}, missing_usage={missing_usage}, "
        "empty_contexts={empty_contexts}".format(**counts)
    )


if __name__ == "__main__":
    main()
