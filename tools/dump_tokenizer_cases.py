#!/usr/bin/env python3
"""Dump reference encodings from HF `tokenizers` for tests/unit/test_tokenizer.cpp.

Usage:
    python tools/dump_tokenizer_cases.py --model models/gpt2 --out tests/data/tokenizer/gpt2.json

Writes {"cases": [{"text": ..., "ids": [...]}, ...], "chats": [...]}:
  cases: hand-picked edge cases plus seeded random strings mixing scripts, digits,
         punctuation, whitespace, emoji and special tokens. Encodings use
         add_special_tokens=False (the engine adds BOS via the chat template).
  chats: conversations rendered by transformers' apply_chat_template (empty if the model has
         no chat template), with the date pinned to DATE_STRING.
"""
import argparse
import json
import random
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer

DATE_STRING = "26 Jul 2024"

CHATS = [
    ([{"role": "user", "content": "Hello"}], True),
    ([{"role": "system", "content": "  Be brief.\n"},
      {"role": "user", "content": "Hi there\n"},
      {"role": "assistant", "content": " Hello! How can I help? "},
      {"role": "user", "content": "What is 2+2?"}], True),
    ([{"role": "system", "content": "You are a pirate."},
      {"role": "user", "content": "Ahoy"},
      {"role": "assistant", "content": "Arr!"}], False),
    ([{"role": "user", "content": "first"},
      {"role": "system", "content": "a mid-conversation system note"},
      {"role": "user", "content": "naïve 你好 😀 <b>tags</b>"}], True),
    ([{"role": "system", "content": "Only a system prompt."}], True),
]


def render_chats(model_dir: Path) -> list[dict]:
    from transformers import AutoTokenizer  # heavy import; only needed here

    hf = AutoTokenizer.from_pretrained(model_dir)
    if not hf.chat_template:
        return []
    return [{"messages": msgs, "add_generation_prompt": gen,
             "text": hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen,
                                            date_string=DATE_STRING)}
            for msgs, gen in CHATS]


HAND_PICKED = [
    "",
    "The capital of France is",
    "Hello, world!",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\r\nand\n\n\nblank lines\n",
    "a  b   c    d\n  e",
    "I'm you're we've he'll they'd it's I'M YOU'RE He'S",
    "x'ſa x'Sa x'LLo x'VEry don'T",  # (?i:...) contractions, incl. long s folding to s
    "1 12 123 1234 12345 3.14159 -42 1,000,000 0x1F 2026-11-13",
    "def f(x):\n    return x**2  # square\n\n\tif x != 0:\n\t\tpass",
    "naïve café résumé Ångström Straße",
    "Привет, мир! Γειά σου κόσμε",
    "你好，世界。日本語のテキスト、한국어 텍스트",
    "مرحبا بالعالم हिन्दी पाठ",
    "emoji 😀👍🏽 👨‍👩‍👧 🇫🇷 ✨",
    "superscripts x² y³ ½ Ⅻ ①②③ ٣٤٥ ４２",
    "non breaking　ideographic em space",
    "!!!??? ... --- *** ''' \"\"\" ``` ~~~",
    "url https://example.com/path?q=1&r=two#frag",
    "x" * 300,
    " " * 50 + "after many spaces",
    "\n" * 20,
]

ALPHABET = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("0123456789")
    + list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    + [" "] * 8 + ["\t", "\n", "\r", " ", "　"]
    + list("éèêëüöäßñçøåÉÜÑ")
    + list("αβγδΩπλжзийДЖ")
    + list("中文字日本語한국")
    + list("ابتثहिन्दी")  # includes combining marks (Mn), which are neither L nor N
    + ["😀", "👍", "🏽", "‍", "🇫", "🇷"]
    + list("²³½Ⅻ①٣４")
    + ["'s", "'t", "'re", "'LL", "'d"]
)


def random_cases(rng: random.Random, n: int) -> list[str]:
    out = []
    for _ in range(n):
        length = rng.choice([1, 2, 5, 10, 30, 80])
        out.append("".join(rng.choice(ALPHABET) for _ in range(length)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model directory with tokenizer.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--random", type=int, default=300, help="number of random strings")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    path = Path(args.model) / "tokenizer.json"
    tok = Tokenizer.from_file(str(path))
    spec = json.loads(path.read_text())
    nfc = (spec.get("normalizer") or {}).get("type") == "NFC"

    rng = random.Random(args.seed)
    specials = [t["content"] for t in spec["added_tokens"]][:8]
    texts = list(HAND_PICKED)
    # Special tokens embedded in ordinary text, adjacent to each other, and at the edges.
    texts += [f"{s}hello {s} world{s}" for s in specials]
    texts.append("".join(specials))
    texts += random_cases(rng, args.random)
    # Random text with special tokens spliced in.
    for t in random_cases(rng, 20):
        k = rng.randrange(len(t) + 1)
        texts.append(t[:k] + rng.choice(specials) + t[k:])

    if nfc:
        # The engine does not implement NFC yet; it assumes input is already NFC.
        texts = [t for t in texts if unicodedata.normalize("NFC", t) == t]

    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in texts]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chats = render_chats(Path(args.model))
    out.write_text(json.dumps({"tokenizer": path.parent.name, "cases": cases, "chats": chats},
                              ensure_ascii=False, indent=0) + "\n")
    print(f"wrote {len(cases)} cases and {len(chats)} chats to {out}")


if __name__ == "__main__":
    main()
