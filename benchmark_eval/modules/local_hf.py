import os
import time
from functools import lru_cache


def _model_id(model: str) -> str:
    for prefix in ("local_hf/", "hf_local/", "local_hf_lora/"):
        if model.startswith(prefix):
            model = model[len(prefix):]
            break
    if "::" in model:
        model = model.split("::", 1)[0]
    return model


def _adapter_id(model: str) -> str | None:
    if model.startswith("local_hf_lora/") and "::" in model:
        return model.split("::", 1)[1]
    return None


@lru_cache(maxsize=2)
def _load_model(model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = _model_id(model)
    adapter_id = _adapter_id(model)
    allow_cpu = os.getenv("LOCAL_HF_ALLOW_CPU", "").lower() in {"1", "true", "yes"}
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "CUDA is not available for local_hf generation. Run on a GPU node, "
            "or set LOCAL_HF_ALLOW_CPU=1 for a slow CPU smoke test."
        )

    token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    trust_remote_code = os.getenv("LOCAL_HF_TRUST_REMOTE_CODE", "1").lower() not in {"0", "false", "no"}
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    model_obj = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
    )
    if adapter_id:
        from peft import PeftModel

        model_obj = PeftModel.from_pretrained(model_obj, adapter_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_obj.to(device)
    model_obj.eval()
    if os.getenv("LOCAL_HF_VERBOSE", "").lower() in {"1", "true", "yes"}:
        print(
            f"[local_hf] loaded {model_id}"
            + (f" + adapter {adapter_id}" if adapter_id else "")
            + f" on {device}"
            + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""),
            flush=True,
        )
    return tokenizer, model_obj, device


def _max_context_length(tokenizer, model_obj) -> int | None:
    candidates = [
        getattr(model_obj.config, "max_position_embeddings", None),
        getattr(model_obj.config, "n_positions", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    valid = [int(c) for c in candidates if isinstance(c, int) and 0 < c < 1_000_000]
    return min(valid) if valid else None


def generate_local_hf(model: str, prompt: str, max_new_tokens: int | None = None) -> str:
    import torch

    tokenizer, model_obj, device = _load_model(model)
    requested_max_new = max_new_tokens or int(os.getenv("LOCAL_HF_MAX_NEW_TOKENS", "2048"))
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        if torch.is_tensor(encoded):
            input_ids = encoded.to(device)
            attention_mask = torch.ones_like(input_ids)
        else:
            if isinstance(encoded, dict) or hasattr(encoded, "__getitem__"):
                input_ids = encoded["input_ids"]
                attention_mask = encoded.get("attention_mask") if hasattr(encoded, "get") else None
            else:
                input_ids = encoded.input_ids
                attention_mask = getattr(encoded, "attention_mask", None)
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    max_context = _max_context_length(tokenizer, model_obj)
    if max_context is not None:
        available = max_context - input_ids.shape[-1]
        if available <= 0:
            raise RuntimeError(
                f"Prompt has {input_ids.shape[-1]} tokens, exceeding local model context "
                f"length {max_context} for {_model_id(model)}."
            )
        max_new_tokens = min(requested_max_new, available)
    else:
        max_new_tokens = requested_max_new

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model_obj.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if os.getenv("LOCAL_HF_VERBOSE", "").lower() in {"1", "true", "yes"}:
        elapsed = time.perf_counter() - start
        print(
            f"[local_hf] generated {new_tokens.shape[-1]} tokens in {elapsed:.1f}s "
            f"({new_tokens.shape[-1] / max(elapsed, 1e-6):.2f} tok/s), "
            f"input_tokens={input_ids.shape[-1]}, max_new_tokens={max_new_tokens}"
            + (f", max_context={max_context}" if max_context else ""),
            flush=True,
        )
    return text
