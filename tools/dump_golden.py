#!/usr/bin/env python3
"""Dump reference tensors from HF transformers for tests/golden.

Usage:
    python tools/dump_golden.py --model /path/to/hf-model --out golden/ \
        --prompt "The capital of France is" --greedy-tokens 32

Writes golden/<name>.safetensors containing:
    input_ids, embeddings, layer_{i}_out (hidden state after each decoder layer),
    final_norm_out, logits_last, greedy_ids
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--greedy-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    # TODO: load with AutoModelForCausalLM, register forward hooks on each decoder layer,
    #       run the prompt, greedy-generate, save all tensors with safetensors.torch.save_file.
    raise NotImplementedError("dump_golden.py")


if __name__ == "__main__":
    main()
