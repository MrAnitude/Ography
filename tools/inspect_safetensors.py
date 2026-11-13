#!/usr/bin/env python3
"""Print the tensor names, dtypes and shapes in a .safetensors file or model directory.

Reads only the JSON header, so it's instant even on multi-GB checkpoints. Handy when
writing the HF-name -> engine-layout table in src/model/weight_map.cpp.

Usage: python tools/inspect_safetensors.py /path/to/model-or-file
"""
import json
import struct
import sys
from pathlib import Path


def read_header(path: Path) -> dict:
    with path.open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(size))


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])
    files = sorted(target.glob("*.safetensors")) if target.is_dir() else [target]
    total = 0
    for file in files:
        print(f"== {file.name}")
        for name, info in sorted(read_header(file).items()):
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            total += end - begin
            print(f"  {name:70s} {info['dtype']:8s} {info['shape']}")
    print(f"total tensor bytes: {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
