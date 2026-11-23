"""Vendored WR KS3 kernels, dense adaptation for ksdraft (2026-11-22).

Source: ~/modded-nanogpt-ks3-pr/track_1_short/ks3/ (the reference
speedrun package), geometry lineage curvedinf/k-splanifolds @ ea7a9c1
(CC-BY-SA-4.0 — see the source checkout's LICENSE.k-splanifolds; ksdraft
is MIT, so this file keeps the provenance note and the user (owner of both
repos) gets to decide the final licensing story).

What this adaptation keeps from the WR package (the pieces our old
ks3_geom.py fused path was missing):

- GS32 activation quantization *during training*: the down-projection
  output and the post-geometry features are rounded onto per-row
  32-channel int8 grids with fp16-rounded scales (``_half_scale``), inside
  the custom-op forward; backward flows through the stored quantized
  tensors.
- In-kernel int8 coefficient grid: c0..c5 round onto a per-chart +/-127
  grid in BOTH the forward geometry and the VJP, so the trained
  coefficients are always consumed as their served grid.
- The nonlinear-mode ladder (``KSDRAFT_KS3_WR_NL``): 2 = int8/int32
  fixed-point Horner with a tanh LUT and Q6 guard bits (the ks3_int8
  lineage), 1 = fp16-stage Horner with fp32 tanh (MI100.md's documented
  fp16-warp regime), 0 = fp32 (parity reference).
- Fused analytic VJP with the 8-way per-chart partial reductions
  (coefficients, scale via sigmoid(raw_scale), shift via tanh').

Dropped (not applicable): MoE expert dispatch/tiled GEMMs (dense draft
decoder — plain torch matmuls), and every H100-only path (FP8 e4/e5, TMA,
h100_kernels, native bit encodings, BF16_TUNED kernels).

Layout mapping to our KS3 module: WR ``up`` [n, d] == our ``down.weight``
(Linear d->features); WR ``down`` [d, n] == our ``up.weight`` (Linear
features->d). Rank-1 charts only, like the decided shape.
"""

from __future__ import annotations

import os

import torch

try:  # Triton is optional (CPU/dev boxes, CI)
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _HAS_TRITON = True
except Exception:  # pragma: no cover - import guard
    _HAS_TRITON = False
    triton = None
    tl = None
    libdevice = None

_LUT: dict[str, torch.Tensor] = {}


def wr_enabled(coords: torch.Tensor) -> bool:
    if not _HAS_TRITON or not coords.is_cuda:
        return False
    if os.environ.get("KSDRAFT_KS3_WR", "1") == "0":
        return False
    return coords.dtype in (torch.float16, torch.bfloat16, torch.float32)


def wr_nl_mode() -> int:
    return int(os.environ.get("KSDRAFT_KS3_WR_NL", "1"))


def tanh_lut(device) -> torch.Tensor:
    """int8 tanh table for NL=2: index ti+128 -> round(127*tanh(ti*3.2/127)).

    ti represents mean*0.25 on the +/-3.2 grid; t is recovered as
    tq*(4/127), matching the WR kernel's fixed-point contract.
    """
    key = str(device)
    if key not in _LUT:
        ti = torch.arange(-128, 129, dtype=torch.float32, device=device)
        lut = torch.round(127.0 * torch.tanh(ti * 3.2 / 127.0))
        _LUT[key] = lut.to(torch.int32)
    return _LUT[key]


if _HAS_TRITON:

    @triton.jit
    def _half_scale(maximum):
        # Round BEFORE quantizing so encoding and decoding use the same FP16 scale.
        return tl.minimum(tl.maximum(maximum / 127., 2.**-24), 65504.).to(tl.float16).to(tl.float32)

    @triton.jit
    def _round_div127(value):
        # Triton integer // truncates toward zero. Correct both signs explicitly.
        quotient = value // 127
        remainder = value - quotient * 127
        return quotient + tl.where(remainder > 63, 1, tl.where(remainder < -63, -1, 0))

    @triton.jit
    def _spine_features(x, mean, c0, c1, c2, c3, c4, c5, ps, scale, shift, LUT,
                        valid, NL: tl.constexpr):
        a = 2*c0-2*c1+c2+c3
        b = -3*c0+3*c1-2*c2-c3
        if NL == 2:
            # INT8 tanh inputs/outputs, INT32 fixed-point Horner accumulators.
            # Q6 guard bits keep worst-case polynomial products within INT32.
            ti = libdevice.nearbyint(tl.minimum(tl.maximum(mean*.25, -3.2), 3.2)*(127./3.2)).to(tl.int32)
            tq = tl.load(LUT+ti+128).to(tl.int32)
            warp = scale[None,:]*(tq.to(tl.float32)*(4./127.)+shift[None,:])
            zi = libdevice.nearbyint(tl.minimum(tl.maximum(warp, -3.2), 3.2)*(127./3.2)).to(tl.int32)
            zq = tl.load(LUT+zi+128).to(tl.int32)
            poly = a[None,:].to(tl.int32)*64
            poly = _round_div127(poly*tq*4)+b[None,:].to(tl.int32)*64
            poly = _round_div127(poly*tq*4)+c2[None,:].to(tl.int32)*64
            poly = _round_div127(poly*tq*4)+c0[None,:].to(tl.int32)*64
            warped = 4*c5[None,:].to(tl.int32)*64
            warped = _round_div127(warped*zq)+2*c4[None,:].to(tl.int32)*64
            warped = _round_div127(warped*zq)-3*c5[None,:].to(tl.int32)*64
            warped = _round_div127(warped*zq)-c4[None,:].to(tl.int32)*64
            h = x+(poly+warped).to(tl.float32)*(ps[None,:]/64.)
            h = tl.where(valid[None,:], h, 0.)
            pre_scale = _half_scale(tl.max(tl.abs(h), 1))
            hq = tl.minimum(tl.maximum(libdevice.nearbyint(h/pre_scale[:, None]), -127.), 127.).to(tl.int32)
            hq = tl.maximum(hq, 0)
            h = (hq*hq).to(tl.float32)*(pre_scale[:, None]*pre_scale[:, None])
        elif NL == 1:
            th = (mean*.25).to(tl.float16)
            t = (4*libdevice.tanh(th.to(tl.float32))).to(tl.float16)
            warp = (scale[None,:]*(t.to(tl.float32)+shift[None,:])).to(tl.float16)
            z = libdevice.tanh(warp.to(tl.float32)).to(tl.float16)
            # Each polynomial stage rounds to FP16; tanh uses FP32 internally.
            poly = (a[None,:].to(tl.float16)*t+b[None,:].to(tl.float16)).to(tl.float16)
            poly = (poly*t+c2[None,:].to(tl.float16)).to(tl.float16)
            poly = (poly*t+c0[None,:].to(tl.float16)).to(tl.float16)
            warped = (4*c5[None,:].to(tl.float16)*z+2*c4[None,:].to(tl.float16)).to(tl.float16)
            warped = (warped*z-3*c5[None,:].to(tl.float16)).to(tl.float16)
            warped = (warped*z-c4[None,:].to(tl.float16)).to(tl.float16)
            h = tl.maximum((x+(poly.to(tl.float32)+warped.to(tl.float32))*ps[None,:]).to(tl.float16), 0.)
            h = (h*h).to(tl.float16).to(tl.float32)
        else:
            t = 4*libdevice.tanh(mean*.25)
            z = libdevice.tanh(scale[None,:]*(t+shift[None,:]))
            poly = ((a[None,:]*t+b[None,:])*t+c2[None,:])*t+c0[None,:]
            warped = ((4*c5[None,:]*z+2*c4[None,:])*z-3*c5[None,:])*z-c4[None,:]
            h = tl.maximum(x+(poly+warped)*ps[None,:], 0.)
            h = h*h
        return tl.where(valid[None,:], h, 0.)

    @triton.jit
    def _pack(A, O, R: tl.constexpr, C: tl.constexpr,
              ROWS_PER_TILE: tl.constexpr = 8, GS32: tl.constexpr = True):
        row = tl.program_id(0)*ROWS_PER_TILE+tl.arange(0, ROWS_PER_TILE)
        col = tl.program_id(1)*32+tl.arange(0, 32)
        mask = (row[:, None] < R) & (col[None, :] < C)
        a = tl.load(A+row[:, None]*C+col[None, :], mask, other=0.).to(tl.float32)
        if GS32:
            s = _half_scale(tl.max(tl.abs(a), 1))
            q = tl.minimum(tl.maximum(libdevice.nearbyint(a/s[:, None]), -127.), 127.)
            values = q*s[:, None]
        else:
            values = a
        tl.store(O+row[:, None]*C+col[None, :], values, mask)

    @triton.jit
    def _geometry(C, G, H, LUT, M: tl.constexpr, N: tl.constexpr,
                  NL: tl.constexpr, BM: tl.constexpr = 16,
                  GS32: tl.constexpr = True):
        rows = tl.program_id(0)*BM+tl.arange(0, BM)
        cols = tl.program_id(1)*32+tl.arange(0, 32)
        valid = cols < N
        base = cols*8
        gp = G+base
        c0 = tl.load(gp+0, valid, other=0.)
        c1 = tl.load(gp+1, valid, other=0.)
        c2 = tl.load(gp+2, valid, other=0.)
        c3 = tl.load(gp+3, valid, other=0.)
        c4 = tl.load(gp+4, valid, other=0.)
        c5 = tl.load(gp+5, valid, other=0.)
        amax = tl.maximum(tl.maximum(tl.abs(c0), tl.abs(c1)),
                          tl.maximum(tl.maximum(tl.abs(c2), tl.abs(c3)),
                                     tl.maximum(tl.abs(c4), tl.abs(c5))))
        ps = _half_scale(amax)
        c0 = libdevice.nearbyint(c0/ps)
        c1 = libdevice.nearbyint(c1/ps)
        c2 = libdevice.nearbyint(c2/ps)
        c3 = libdevice.nearbyint(c3/ps)
        c4 = libdevice.nearbyint(c4/ps)
        c5 = libdevice.nearbyint(c5/ps)
        c0 = tl.minimum(tl.maximum(c0, -127.), 127.)
        c1 = tl.minimum(tl.maximum(c1, -127.), 127.)
        c2 = tl.minimum(tl.maximum(c2, -127.), 127.)
        c3 = tl.minimum(tl.maximum(c3, -127.), 127.)
        c4 = tl.minimum(tl.maximum(c4, -127.), 127.)
        c5 = tl.minimum(tl.maximum(c5, -127.), 127.)
        raw = tl.load(G+base+6, valid, other=0.)
        rh = tl.load(G+base+7, valid, other=0.)
        scale = tl.where(raw > 20., raw, libdevice.log1p(libdevice.exp(raw)))+1e-4
        shift = .5*libdevice.tanh(rh)
        x = tl.load(C+rows[:, None]*N+cols[None, :],
                    (rows[:, None] < M) & valid[None, :], other=0.)
        x = tl.minimum(tl.maximum(x, -65504.), 65504.)
        h = _spine_features(x, x, c0, c1, c2, c3, c4, c5, ps, scale, shift,
                            LUT, valid, NL)
        if GS32:
            hs = _half_scale(tl.max(tl.abs(h), 1))
            q = tl.minimum(tl.maximum(libdevice.nearbyint(h/hs[:, None]), -127.), 127.)
            values = q*hs[:, None]
        else:
            values = h
        tl.store(H+rows[:, None]*N+cols[None, :], values,
                 (rows[:, None] < M) & valid[None, :])

    @triton.jit
    def _vjp(V, C, G, DC, PART, LUT, M: tl.constexpr, N: tl.constexpr,
             NL: tl.constexpr, BM: tl.constexpr = 64, GS32: tl.constexpr = True):
        rows = tl.program_id(0)*BM+tl.arange(0, BM)
        cols = tl.program_id(1)*32+tl.arange(0, 32)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        value = tl.load(V+rows[:, None]*N+cols[None, :], mask, other=0.)
        original = tl.load(C+rows[:, None]*N+cols[None, :], mask, other=0.)
        x = tl.minimum(tl.maximum(original, -65504.), 65504.)
        base = cols*8
        gp = G+base
        c0 = tl.load(gp, cols < N, other=0.)
        c1 = tl.load(gp+1, cols < N, other=0.)
        c2 = tl.load(gp+2, cols < N, other=0.)
        c3 = tl.load(gp+3, cols < N, other=0.)
        c4 = tl.load(gp+4, cols < N, other=0.)
        c5 = tl.load(gp+5, cols < N, other=0.)
        amax = tl.maximum(tl.maximum(tl.abs(c0), tl.abs(c1)),
                          tl.maximum(tl.maximum(tl.abs(c2), tl.abs(c3)),
                                     tl.maximum(tl.abs(c4), tl.abs(c5))))
        ps = _half_scale(amax)
        c0 = tl.minimum(tl.maximum(libdevice.nearbyint(c0/ps), -127.), 127.)*ps
        c1 = tl.minimum(tl.maximum(libdevice.nearbyint(c1/ps), -127.), 127.)*ps
        c2 = tl.minimum(tl.maximum(libdevice.nearbyint(c2/ps), -127.), 127.)*ps
        c3 = tl.minimum(tl.maximum(libdevice.nearbyint(c3/ps), -127.), 127.)*ps
        c4 = tl.minimum(tl.maximum(libdevice.nearbyint(c4/ps), -127.), 127.)*ps
        c5 = tl.minimum(tl.maximum(libdevice.nearbyint(c5/ps), -127.), 127.)*ps
        rs = tl.load(G+base+6, cols < N, other=0.)
        rh = tl.load(G+base+7, cols < N, other=0.)
        scale = tl.where(rs > 20., rs, libdevice.log1p(libdevice.exp(rs)))+1e-4
        shift = .5*libdevice.tanh(rh)
        t = 4.*libdevice.tanh(x*.25)
        z = libdevice.tanh(scale[None, :]*(t+shift[None, :]))
        cubic = 2*c0-2*c1+c2+c3
        quadratic = -3*c0+3*c1-2*c2-c3
        spine = ((cubic*t+quadratic)*t+c2)*t+c0
        warped = ((4*c5*z+2*c4)*z-3*c5)*z-c4
        dp = tl.where(mask, value*2.*tl.maximum(x+spine+warped, 0.), 0.)
        dz = ((12*c5*z+4*c4)*z-3*c5)*(1.-z*z)
        dm = dp*((3*cubic*t+2*quadratic)*t+c2+dz*scale)*(1.-t*t*.0625)
        dc = tl.where((original >= -65504.) & (original <= 65504.), dp, 0.)
        tl.store(DC+rows[:, None]*N+cols[None, :], dc, mask)
        off = (tl.program_id(0)*N+cols)*8
        t2 = t*t
        t3 = t2*t
        tl.store(PART+off, tl.sum(dp*(2*t3-3*t2+1), 0), cols < N)
        tl.store(PART+off+1, tl.sum(dp*(-2*t3+3*t2), 0), cols < N)
        tl.store(PART+off+2, tl.sum(dp*(t3-2*t2+t), 0), cols < N)
        tl.store(PART+off+3, tl.sum(dp*(t3-t2), 0), cols < N)
        tl.store(PART+off+4, tl.sum(dp*(2*z*z-1), 0), cols < N)
        tl.store(PART+off+5, tl.sum(dp*(4*z*z*z-3*z), 0), cols < N)
        tl.store(PART+off+6, tl.sum(dp*dz*(t+shift)*(1.0/(1.0+tl.exp(-rs))), 0), cols < N)
        tl.store(PART+off+7, tl.sum(dp*dz*scale*.5*(1.-libdevice.tanh(rh)*libdevice.tanh(rh)), 0), cols < N)

    @triton.jit
    def _reduce(P, O, N: tl.constexpr, T: tl.constexpr, B: tl.constexpr):
        n = tl.program_id(0)
        r = tl.arange(0, B)
        j = tl.arange(0, 8)
        v = tl.load(P+(r[:, None]*N+n)*8+j[None, :], r[:, None] < T, other=0.)
        total = tl.sum(v, 0)
        tl.store(O+n*8+j, total)


def _pack_quantize(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.float32)
    m, n = x.shape
    _pack[(triton.cdiv(m, 8), triton.cdiv(n, 32))](x, out, m, n, num_warps=4)
    return out.to(x.dtype).contiguous()


def _geometry_pack(g: torch.Tensor) -> torch.Tensor:
    """(charts, 6) + raw_scale + raw_shift -> contiguous (charts, 8) buffer."""
    return torch.cat([g["coefficients"].float(),
                      g["raw_scale"].float()[:, None],
                      g["raw_shift"].float()[:, None]], dim=1).contiguous()


class _KS3WR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, down_w, up_w, coefficients, raw_scale, raw_shift,
                nl: int, gs32: bool):
        # WR "up" [n, d] == our down.weight; WR "down" [d, n] == our up.weight
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        xq = _pack_quantize(x.detach().float())
        c = xq @ down_w.detach().float().t()          # [M, n]
        g = {"coefficients": coefficients.detach(),
             "raw_scale": raw_scale.detach(), "raw_shift": raw_shift.detach()}
        G = _geometry_pack(g)
        M, N = c.shape
        h = torch.empty((M, N), device=x.device, dtype=torch.float32)
        nl_mode = nl if nl in (0, 1, 2) else 1
        _geometry[(triton.cdiv(M, 16), triton.cdiv(N, 32))](
            c.contiguous(), G, h, tanh_lut(x.device), M, N,
            NL=nl_mode, GS32=gs32, num_warps=4)
        # GS32 quantization of the features already happened in-kernel
        y = h.to(up_w.dtype) @ up_w.detach().t()
        ctx.save_for_backward(xq, c, h, down_w, up_w, G)
        ctx.nl, ctx.gs32 = nl_mode, gs32
        return y.view(*orig_shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, grad):
        xq, c, h, down_w, up_w, G = ctx.saved_tensors
        M, N = c.shape
        gg = grad.detach().reshape(-1, grad.shape[-1]).float()
        v = gg @ up_w.detach().float()               # [M, n]
        dc = torch.empty((M, N), device=grad.device, dtype=torch.float32)
        part = torch.empty((triton.cdiv(M, 64), N, 8), device=grad.device,
                           dtype=torch.float32)
        _vjp[(triton.cdiv(M, 64), triton.cdiv(N, 32))](
            v.contiguous(), c, G, dc, part, tanh_lut(grad.device), M, N,
            NL=ctx.nl, GS32=False, num_warps=4)
        dg = torch.empty((N, 8), device=grad.device, dtype=torch.float32)
        T = part.shape[0]
        _reduce[(N,)](part, dg, N, T, triton.next_power_of_2(T) if T > 1 else 1)
        dx = dc @ down_w.detach().float()            # [M, d]
        d_down = torch.mm(dc.t().to(h.dtype), xq).to(down_w.dtype)   # [n, d]
        d_up = torch.mm(gg.t().to(h.dtype), h).to(up_w.dtype)        # [d, n]
        return dx.view(grad.shape).to(grad.dtype), d_down, d_up, \
            dg[:, :6].contiguous(), dg[:, 6].contiguous(), dg[:, 7].contiguous(), \
            None, None


def ks3_wr_forward(x: torch.Tensor, module, nl: int | None = None,
                   gs32: bool = True):
    """Drop-in fast path for KS3.forward using the vendored WR kernels."""
    nl = wr_nl_mode() if nl is None else nl
    return _KS3WR.apply(
        x, module.down.weight, module.up.weight, module.coefficients,
        module.raw_scale, module.raw_shift, int(nl), bool(gs32))
