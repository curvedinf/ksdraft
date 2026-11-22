"""Convert a legacy fp16-container signal store to the v2 int8 codes+scales
form (the cache layout itself). Used to smoke-test the v2 loader/trainer on
already-mined data before the fork-side deep miner produces native v2 stores.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/curved/ksdraft")
from ksdraft.teacher import blocks_int8_kv


def convert(src: Path, dst: Path, tokens: int) -> None:
    meta = json.loads((src / "meta.json").read_text())
    taps = len(meta["taps"])
    rows = min(int(meta["tokens"]), tokens)
    dst.mkdir(parents=True, exist_ok=True)
    for stem in ("k", "v"):
        st = torch.from_file(str(src / f"{stem}_stream.bin"), shared=False,
                             size=int(meta["tokens"]) * taps * 1024,
                             dtype=torch.float16).view(-1, taps, 1024)
        cf = open(dst / f"{stem}_codes.bin", "wb")
        sf = open(dst / f"{stem}_scales.bin", "wb")
        step = 4096
        for lo in range(0, rows, step):
            blk = st[lo:min(lo + step, rows)].float()
            c, s = blocks_int8_kv(blk)
            cf.write(c.numpy().tobytes())
            sf.write(s.numpy().tobytes())
        cf.close()
        sf.close()
        print(f"{stem}: {rows} rows -> codes+scales")
    ti = torch.load(src / "topk_ids.pt", weights_only=True)[:rows]
    tp = torch.load(src / "topk_probs.pt", weights_only=True)[:rows]
    torch.save(ti, dst / "topk_ids.pt")
    torch.save(tp, dst / "topk_probs.pt")
    b = torch.load(src / "boundaries.pt", weights_only=True)
    torch.save(b[b <= rows], dst / "boundaries.pt")
    (dst / "meta.json").write_text(json.dumps({
        "source": meta.get("source"), "corpus": meta.get("corpus"),
        "tokens": rows, "taps": meta["taps"], "topk": meta["topk"],
        "format": "ksdraft-kv-int8-v2",
        "kv_grid": "int8_block_g128 codes+scales (fork kernel semantics)",
        "codes_dtype": "int8", "scales_dtype": "fp16", "group": 128,
        "converted_from": str(src)}, indent=2))
    print(f"wrote {dst} ({rows} rows)")


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
