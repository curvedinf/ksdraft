# SPDX-License-Identifier: Apache-2.0
"""ksdraft: single-file inference for the DFly2-style target-KV reuse drafter.

The draft is an N-layer (4 in the shipped checkpoint) block-drafting trunk:
Q attends the target's cached K/V windows (no drafter KV cache), a KS3 spine
decoder mixes each layer, and DSpine injections condition on adjacent
positions. The vocabulary embedding and output head are the *target's*
tables, bound externally at load.

    import ksdraft, torch
    draft = ksdraft.from_pretrained(
        "models/inkling-2-mini-ksdraft",
        embedding=target.get_input_embeddings().weight,   # [V, d]
        head=target.get_output_embeddings().weight,       # [V, d]
        device="cuda", dtype="fp16")
    # int8 KV cache taps: (k_codes, k_scales, v_codes, v_scales), each
    # [B, layers, S, kv_dim] (scales [B, layers, S, kv_dim/128]) or pass
    # float (k, v) streams instead.
    toks = draft.propose(tokens, n=16, target_kv=(kc, ks, vc, vs))

This is the portable reference path: int8 weights load as codes x scales,
activations quantize to the same int8 g128 grid, and the GEMMs use the
reference dequantized-operand fp32 epilogue (no custom kernels). KS3 uses
the canonical Horner geometry instead of the int8 tanh-LUT nonlinearity.
The artifact ships no transfer table, so the last-write refinement is the
identity and is omitted.

Packaging: publish this file as `ksdraft.py` (py_modules=["ksdraft"]); the
only runtime dependency is torch (safetensors optional, torch load fallback).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

__version__ = "0.1.0"
__all__ = ["KSDraftConfig", "KSDraft", "from_pretrained", "load"]

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
_GROUP = 128


@dataclass
class KSDraftConfig:
    vocab_size: int = 201024
    layers: int = 4
    d_model: int = 2304
    head_size: int = 64
    ks3_charts: int = 24800
    block_len: int = 16
    kv_dim: int = 1024
    n_kv_heads: int = 8
    kv_window: int = 1024
    tie_word_embeddings: bool = False

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            setattr(self, f.name,
                    bool(v) if isinstance(v, bool) else int(v))

    @classmethod
    def from_json(cls, path, **overrides) -> "KSDraftConfig":
        raw = json.loads((Path(path) / "config.json").read_text())["config"]
        names = {f.name for f in fields(cls)}
        return cls(**{**{k: v for k, v in raw.items() if k in names}, **overrides})

    @property
    def n_kv(self) -> int:
        return max(1, min(self.n_kv_heads, self.d_model // self.head_size))

    @property
    def n_q(self) -> int:
        base = self.d_model // self.head_size
        return max(self.n_kv, (base // self.n_kv) * self.n_kv)


def _deq(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """int8 codes x per-group scales (group 128 along the last dim)."""
    k = codes.shape[-1]
    kp = (k + _GROUP - 1) // _GROUP * _GROUP
    if kp != k:
        codes = F.pad(codes, (0, kp - k))
    c = codes.reshape(*codes.shape[:-1], kp // _GROUP, _GROUP).float()
    w = c * scales.float()[..., None]
    return w.reshape(*codes.shape[:-1], kp)[..., :k]


def _quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 g128 quantizer (same grid as the checkpoint)."""
    x = x.float()
    k = x.shape[-1]
    kp = (k + _GROUP - 1) // _GROUP * _GROUP
    if kp != k:
        x = F.pad(x, (0, kp - k))
    g = x.reshape(*x.shape[:-1], kp // _GROUP, _GROUP)
    s = (g.abs().amax(-1) / 127.0).clamp(2.0**-24, 65504.0)
    s = s.half().float()                       # round through fp16, use fp32
    c = torch.clamp(torch.round(g / s[..., None]), -127.0, 127.0)
    return c.to(torch.int8).reshape(*x.shape[:-1], kp)[..., :k].contiguous(), \
        s.half()


def _quant_rows(x: torch.Tensor, chunk: int = 8192):
    """Row-chunked int8 quantization (bounded fp32 transients)."""
    codes, scales = [], []
    for i in range(0, x.shape[0], chunk):
        c, s = _quant(x[i:i + chunk])
        codes.append(c)
        scales.append(s)
    return torch.cat(codes), torch.cat(scales)


def _ln(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)


def _load_sd(path: Path) -> dict:
    st = path / "ksdraft.safetensors"
    if st.exists():
        from safetensors.torch import load_file

        return dict(load_file(str(st)))
    return dict(torch.load(path / "ksdraft.pt", map_location="cpu"))


class KSDraft:
    """Inference-only drafter. Built by :func:`from_pretrained`."""

    def __init__(self, cfg: KSDraftConfig, sd: dict, embedding, head,
                 dtype=torch.float16, device="cpu"):
        self.config = cfg
        self.dtype = dtype
        self.device = torch.device(device)

        def t(x):
            return x.to(device=self.device, dtype=dtype)

        def w(name):
            return t(_deq(sd[name + ".weight_codes"], sd[name + ".weight_scales"]))

        def d(name):
            return t(_deq(sd[name + "_codes"], sd[name + "_scales"]))

        emb = embedding if embedding is not None else sd.get("embedding.weight")
        hd = head if head is not None else sd.get("head.weight")
        if hd is None and cfg.tie_word_embeddings:
            hd = emb
        if emb is None or hd is None:
            raise ValueError("pass the target's embedding= and head= tables "
                             "(this artifact ships no vocabulary table)")
        if tuple(emb.shape) != (cfg.vocab_size, cfg.d_model):
            raise ValueError(f"embedding {tuple(emb.shape)} != "
                             f"{(cfg.vocab_size, cfg.d_model)}")
        if tuple(hd.shape) != (cfg.vocab_size, cfg.d_model):
            raise ValueError(f"head {tuple(hd.shape)} != "
                             f"{(cfg.vocab_size, cfg.d_model)}")
        self.embedding = emb.to(self.device)
        self.head_codes, self.head_scales = _quant_rows(hd)
        self.head_codes = self.head_codes.to(self.device)
        self.head_scales = self.head_scales.to(self.device)
        self.input_norm = (t(sd["input_norm.weight"]), t(sd["input_norm.bias"]))
        self.output_norm = (t(sd["output_norm.weight"]), t(sd["output_norm.bias"]))
        self.A = t(sd["spine_core.A.weight"])
        self.Gq = t(sd["spine_core.Gq.weight"])
        self.Gp = t(sd["spine_core.Gp.weight"])
        self.spine_b = t(sd["spine_core.b"])
        self.bc_w = t(sd["block_conv.dynamic.weight"])
        self.bc_b = t(sd["block_conv.base"])
        self.block_delta = t(sd["block_delta"])

        self.blocks = []
        for i in range(cfg.layers):
            p = f"blocks.{i}."
            b = SimpleNamespace(
                ln1=(t(sd[p + "ln_attn.weight"]), t(sd[p + "ln_attn.bias"])),
                ln2=(t(sd[p + "ln_ffn.weight"]), t(sd[p + "ln_ffn.bias"])),
                q=w(p + "attn.q_proj"), k=w(p + "attn.k_adapter"),
                v=w(p + "attn.v_adapter"), o=w(p + "attn.out_proj"),
                down=d(p + "ffn.down"), up=d(p + "ffn.up"),
                coeffs=t(sd[p + "ffn.spine_codes"].float()
                         * sd[p + "ffn.spine_scale"].float()[:, None]),
                ws=t(_deq(sd[p + "ffn.warp_scale_codes"],
                          sd[p + "ffn.warp_scale_scales"]).reshape(-1)),
                wsh=t(_deq(sd[p + "ffn.warp_shift_codes"],
                           sd[p + "ffn.warp_shift_scales"]).reshape(-1)),
                R=t(sd[p + "injection.R.weight"]),
                W=t(sd[p + "injection.W.weight"]),
            )
            self.blocks.append(b)

    # ---- a target-KV tap -> per-layer float streams ----

    def _kv(self, target_kv):
        cfg = self.config
        if target_kv is None:
            raise ValueError("this reuse drafter needs target KV at every "
                             "forward: pass int8 codes/scales or float streams")
        if len(target_kv) == 4:
            kc, ks, vc, vs = target_kv
            out = []
            for i in range(cfg.layers):
                out.append((_deq(kc[:, i], ks[:, i]).to(self.dtype),
                            _deq(vc[:, i], vs[:, i]).to(self.dtype)))
            return out
        k, v = target_kv[:2]
        return [(k[:, i].to(self.dtype), v[:, i].to(self.dtype))
                for i in range(cfg.layers)]

    # ---- layers ----

    def _attn(self, x, b, ksrc, vsrc):
        B, T, _ = x.shape
        hs, nkv, nq = self.config.head_size, self.config.n_kv, self.config.n_q
        q = F.linear(_ln(x, *b.ln1), b.q).view(B, T, nkv, nq // nkv, hs)
        k = F.linear(ksrc, b.k).view(B, -1, nkv, hs)
        v = F.linear(vsrc, b.v).view(B, -1, nkv, hs)
        S = k.shape[1]
        s = torch.einsum("btkgh,bskh->btkgs", q.float(), k.float()) / hs**0.5
        off, win = S - T, S - T + 1
        qp = torch.arange(T, device=x.device) + off
        kp = torch.arange(S, device=x.device)
        mask = (kp[None, :] <= qp[:, None]) & (kp[None, :] > qp[:, None] - win)
        s = s.masked_fill(~mask[None, :, None, None, :], float("-inf"))
        o = torch.einsum("btkgs,bskh->btkgh", s.softmax(-1), v.float())
        o = o.reshape(B, T, nq * hs)
        return F.linear(o.to(x.dtype), b.o)

    def _ks3(self, x, b):
        coords = F.linear(x, b.down).float().clamp(-65504.0, 65504.0)
        h = coords.reshape(*x.shape[:-1], self.config.ks3_charts, 1)
        tt = 4.0 * torch.tanh(h / 4.0)
        z = torch.tanh(b.ws.float()[:, None] * (tt + b.wsh.float()[:, None]))
        p0, p1, p2, p3, p4, p5 = b.coeffs.float().reshape(-1, 1, 6).unbind(-1)
        cubic = 2 * p0 - 2 * p1 + p2 + p3
        quad = -3 * p0 + 3 * p1 - 2 * p2 - p3
        spine = ((cubic * tt + quad) * tt + p2) * tt + p0
        warp = ((4 * p5 * z + 2 * p4) * z - 3 * p5) * z - p4
        f = F.relu(h + spine + warp).square().flatten(-2)
        return F.linear(f.to(x.dtype), b.up)

    def _inject(self, u, b):
        z = math.sqrt(self.config.d_model) * F.normalize(
            F.linear(u.float(), b.R.float()), dim=-1)
        zp = torch.zeros_like(z)
        zp[:, 1:] = z[:, :-1]
        g = torch.sigmoid(z @ self.Gq.float().T
                          + zp @ self.Gp.float().T + self.spine_b.float())
        m = zp @ self.A.float().T
        rms = (u.float().pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        return (u.float() + rms * (g * m) @ b.W.float().T).to(u.dtype)

    def _block_conv(self, hs):
        B, n, d = hs.shape
        taps = self.bc_b.shape[0]
        ker = F.linear(hs, self.bc_w).view(B, n, taps, d)
        g = 16
        if d % g == 0:
            ker = (ker.view(B, n, taps, d // g, g).mean(-1, keepdim=True)
                   .expand(B, n, taps, d // g, g).reshape(B, n, taps, d))
        out = (self.bc_b[0] + ker[:, :, 0]) * hs
        for j in range(1, taps):
            sh = torch.zeros_like(hs)
            sh[:, j:] = hs[:, :-j]
            out = out + (self.bc_b[j] + ker[:, :, j]) * sh
        return out

    def _lm(self, h):
        """Head logits on the int8 g128 grid (fp32 epilogue), chunked."""
        shape = h.shape[:-1]
        h = h.reshape(-1, h.shape[-1]).to(self.dtype)
        ac, asc = _quant(h)
        a = _deq(ac, asc)
        n = self.head_codes.shape[0]
        step = max(1, 16_000_000 // self.head_codes.shape[1])
        y = torch.cat([a @ _deq(self.head_codes[i:i + step],
                                self.head_scales[i:i + step]).T
                       for i in range(0, n, step)], -1)
        return y.reshape(*shape, n)

    # ---- public API ----

    def trunk(self, tokens, target_kv=None):
        tokens = tokens.to(self.device)
        x = _ln(self.embedding[tokens].to(self.dtype), *self.input_norm)
        for b, (k, v) in zip(self.blocks, self._kv(target_kv)):
            h = x + self._attn(x, b, k, v)
            x = h + self._ks3(_ln(h, *b.ln2), b)
            x = self._inject(x, b)
        return _ln(x, *self.output_norm)

    @torch.no_grad()
    def forward(self, tokens, target_kv=None):
        """tokens [B, T] -> logits [B, T, V] for the next-token distributions."""
        return self._lm(self.trunk(tokens, target_kv))

    @torch.no_grad()
    def block_hidden(self, tokens, n=16, target_kv=None):
        if n > self.block_delta.shape[0]:
            raise ValueError(f"n={n} exceeds block_len="
                             f"{self.block_delta.shape[0]}")
        h = self.trunk(tokens, target_kv)
        return self._block_conv(h[:, -1:] + self.block_delta[:n])

    @torch.no_grad()
    def block_logits(self, tokens, n=16, target_kv=None):
        """One pass -> [B, n, V]; offset k predicts token anchor+k+1."""
        return self._lm(self.block_hidden(tokens, n, target_kv))

    @torch.no_grad()
    def propose(self, tokens, n=16, target_kv=None, topk=16,
                return_conf=False):
        """Greedy block proposal -> tokens [B, n] (and confidences)."""
        logits = self.block_logits(tokens, n, target_kv)
        vals, ids = logits.topk(topk, dim=-1)
        pick = vals.argmax(-1, keepdim=True)
        toks = ids.gather(-1, pick).squeeze(-1)
        if return_conf:
            conf = vals.softmax(-1).gather(-1, pick).squeeze(-1)
            return toks, conf
        return toks


def from_pretrained(path, embedding=None, head=None, device="cpu",
                    dtype="fp16", **config_overrides) -> KSDraft:
    """Load an exported draft directory.

    ``dtype``: "fp16" (default), "fp32", "bf16", or a torch dtype.
    ``embedding``/``head``: the target's [vocab, d_model] tables (required
    for a shared-table artifact; otherwise read from the checkpoint).
    ``**config_overrides``: e.g. ``kv_window=2048``.
    """
    dt = _DTYPES[dtype] if isinstance(dtype, str) else dtype
    cfg = KSDraftConfig.from_json(path, **config_overrides)
    sd = _load_sd(Path(path))
    return KSDraft(cfg, sd, embedding, head, dtype=dt, device=device)


load = from_pretrained
