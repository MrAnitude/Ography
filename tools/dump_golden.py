#!/usr/bin/env python3
"""Dump reference tensors from HF transformers for tests/golden.

Usage:
    python tools/dump_golden.py --model models/gpt2 --out golden/gpt2

Runs each test case through the model on the CPU in fp32 and writes:

    <out>/manifest.json         {"model": <abs model dir>, "cases": [...]}
    <out>/<case>.safetensors    one file per case, all float32 / int32:
        input_ids         [T]           prompt token ids
        hidden_{i}        [T, H]        i = 0: embeddings; i = 1..L: residual stream after
                                        decoder layer i-1 (before the final norm)
        final_norm        [T, H]        final norm output
        logits_last       [V]           logits of the last prompt token
        greedy_ids        [N]           greedy continuation
        greedy_margins    [N]           top1 - top2 logit at each greedy step

The engine side (tests/golden/test_golden.cpp) compares each tensor in order; the first one
that diverges points at the bug.

AWQ / GPTQ checkpoints (quantization_config in config.json) are dequantized here, in plain
torch, into an fp32 model: the reference for the engine's INT4 path. --check-against <fp16
model dir> also compares that model's predictions with the original model's, which catches a
misread packing format (near-zero agreement instead of most positions). Greedy steps whose margin is tiny are ties that fp16 may
legitimately break the other way, so the test checks those leniently.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# A short prompt, plus a long one that spans several prefill chunks and many positions.
DEFAULT_CASES = {
    "short": "The capital of France is",
    "long": (
        "In the beginning of the nineteenth century, the town was a small fishing village "
        "on the northern coast. The harbour was shallow, and the boats were drawn up on the "
        "shingle each evening. "
    )
    * 12
    + "Over the following decades",
}


AWQ_NIBBLE = [0, 4, 1, 5, 2, 6, 3, 7]  # AWQ packs column 8b+j into nibble AWQ_NIBBLE[j]


def unpack_int4(packed: torch.Tensor, dim: int, order=None) -> torch.Tensor:
    """Unpacks int32 words into 8 nibbles each along `dim` (nibble k = bits 4k..4k+3)."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    nib = (packed.unsqueeze(-1) >> shifts) & 0xF  # [..., 8]
    if order is not None:
        nib = nib[..., order]
    if dim == 1:  # [rows, cols/8, 8] -> [rows, cols]
        return nib.reshape(packed.shape[0], -1)
    return nib.permute(0, 2, 1).reshape(-1, packed.shape[1])  # [rows/8, cols, 8] -> [rows, cols]


def dequantize_checkpoint(model_dir: Path, qcfg: dict) -> dict[str, torch.Tensor]:
    """Loads an AWQ ("gemm") or GPTQ (no act-order) checkpoint as fp32 nn.Linear weights."""
    from safetensors import safe_open

    method, group = qcfg["quant_method"], qcfg["group_size"]
    raw = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as st:
            for k in st.keys():
                raw[k] = st.get_tensor(k)
    out = {}
    for name, t in raw.items():
        if name.endswith((".qzeros", ".scales", ".g_idx")):
            continue
        if not name.endswith(".qweight"):
            out[name] = t.float()
            continue
        prefix = name[: -len(".qweight")]
        scales = raw[prefix + ".scales"].float()  # [groups, out]
        if method == "awq":
            q = unpack_int4(t, dim=1, order=AWQ_NIBBLE)                     # [in, out]
            z = unpack_int4(raw[prefix + ".qzeros"], dim=1, order=AWQ_NIBBLE)
        else:
            q = unpack_int4(t, dim=0)                                       # [in, out]
            z = unpack_int4(raw[prefix + ".qzeros"], dim=1) + 1            # stored as zero - 1
        g = torch.arange(q.shape[0]) // group
        w = (q.float() - z[g].float()) * scales[g]                          # [in, out]
        out[prefix + ".weight"] = w.t().contiguous()                        # nn.Linear: [out, in]
    return out


def load_model(model_dir: Path):
    config = json.loads((model_dir / "config.json").read_text())
    qcfg = config.get("quantization_config")
    if not qcfg:
        return AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval(), None
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir)
    del cfg.quantization_config
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()
    state = dequantize_checkpoint(model_dir, qcfg)
    if cfg.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if not (cfg.tie_word_embeddings and m == "lm_head.weight")]
    if missing or unexpected:
        raise RuntimeError(f"dequantized state dict: missing {missing}, unexpected {unexpected}")
    if cfg.tie_word_embeddings:
        model.tie_weights()
    return model, state


@torch.no_grad()
def check_against(model, tokenizer, ref_dir: Path, text: str) -> None:
    """Next-token agreement and KL of the dequantized model vs the original fp16 model.

    Weight-by-weight comparison doesn't work for AWQ, which rescales input channels and folds
    the inverse into the preceding layer. A misread packing gives near-zero agreement; correct
    4-bit weights agree on most positions.
    """
    ref = AutoModelForCausalLM.from_pretrained(ref_dir, dtype=torch.float32).eval()
    ids = tokenizer(text, return_tensors="pt").input_ids
    lq = model(ids).logits[0].log_softmax(-1)
    lr = ref(ids).logits[0].log_softmax(-1)
    agree = float((lq.argmax(-1) == lr.argmax(-1)).float().mean())
    kl = float((lr.exp() * (lr - lq)).sum(-1).mean())
    print(f"vs {ref_dir.name} over {ids.shape[1]} tokens: next-token agreement {agree:.1%}, "
          f"mean KL {kl:.4f}")
    if agree < 0.6:
        raise RuntimeError("dequantized model disagrees with the original: packing misread?")


def decoder_layers(model: torch.nn.Module, n_layers: int) -> torch.nn.ModuleList:
    """Finds the ModuleList holding the decoder layers (transformer.h, model.layers, ...)."""
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == n_layers:
            return module
    raise RuntimeError("could not find the decoder layer list")


@torch.no_grad()
def dump_case(model, tokenizer, prompt: str, greedy_tokens: int) -> dict[str, torch.Tensor]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    n_layers = model.config.num_hidden_layers

    # Capture each decoder layer's output (the residual stream after that layer).
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for i, layer in enumerate(decoder_layers(model, n_layers)):
        def hook(_mod, _inp, out, i=i):
            captured[i] = (out[0] if isinstance(out, tuple) else out)[0].detach().clone()
        hooks.append(layer.register_forward_hook(hook))
    out = model(ids, output_hidden_states=True, use_cache=False)
    for h in hooks:
        h.remove()

    tensors = {"input_ids": ids[0].to(torch.int32)}
    # hidden_states[0] is the embedding output; hidden_states[L] is final-norm output.
    tensors["hidden_0"] = out.hidden_states[0][0]
    for i in range(n_layers):
        tensors[f"hidden_{i + 1}"] = captured[i]
    tensors["final_norm"] = out.hidden_states[n_layers][0]
    tensors["logits_last"] = out.logits[0, -1]

    # Greedy decode step by step so we can record each step's top-2 margin.
    greedy, margins = [], []
    past = None
    step_input = ids
    for _ in range(greedy_tokens):
        o = model(step_input, past_key_values=past, use_cache=True)
        past = o.past_key_values
        top2 = torch.topk(o.logits[0, -1], 2)
        greedy.append(int(top2.indices[0]))
        margins.append(float(top2.values[0] - top2.values[1]))
        step_input = top2.indices[:1].view(1, 1)
    tensors["greedy_ids"] = torch.tensor(greedy, dtype=torch.int32)
    tensors["greedy_margins"] = torch.tensor(margins, dtype=torch.float32)

    return {k: v.contiguous().float() if v.is_floating_point() else v.contiguous()
            for k, v in tensors.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--prompt", help="dump a single case named 'custom' with this prompt")
    parser.add_argument("--greedy-tokens", type=int, default=32)
    parser.add_argument("--check-against", help="original fp16 model dir (quantized models only)")
    args = parser.parse_args()

    model_dir = Path(args.model).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model, dequantized = load_model(model_dir)
    if args.check_against:
        if dequantized is None:
            parser.error("--check-against only applies to quantized models")
        check_against(model, tokenizer, Path(args.check_against), DEFAULT_CASES["long"])

    cases = {"custom": args.prompt} if args.prompt else DEFAULT_CASES
    for name, prompt in cases.items():
        tensors = dump_case(model, tokenizer, prompt, args.greedy_tokens)
        save_file(tensors, str(out_dir / f"{name}.safetensors"))
        text = tokenizer.decode(tensors["greedy_ids"].tolist())
        print(f"{name}: {tensors['input_ids'].numel()} prompt tokens -> {text!r}")

    manifest = {"model": str(model_dir), "cases": list(cases)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out_dir}/manifest.json")


if __name__ == "__main__":
    main()
