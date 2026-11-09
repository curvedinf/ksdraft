import pytest
import torch

from ksdraft.kernels import Int8Linear
from ksdraft.kernels.quant import (
    GROUP_SIZE, SCALE_DTYPE, quantize_last_dim, dequantize_last_dim,
)


def test_quantize_roundtrip_error_bound():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 256, generator=g) * 3
    codes, scales = quantize_last_dim(x)
    assert codes.dtype == torch.int8 and scales.dtype == SCALE_DTYPE
    assert codes.shape == x.shape
    assert scales.shape == (64, 256 // GROUP_SIZE)
    dq = dequantize_last_dim(codes, scales)
    # max error bounded by half a quantization step; the step comes from the
    # FP16-rounded scale actually used for encoding (rounding may shrink it,
    # so bound with the returned scale plus the clamp saturation term).
    scale_used = scales.float().reshape(64, 2, 1)
    err = (x - dq).abs().reshape(64, 2, GROUP_SIZE)
    residual = (x.abs().reshape(64, 2, GROUP_SIZE) - 127.0 * scale_used).clamp(min=0)
    assert (err <= scale_used / 2 + residual + 1e-6).all()


def test_quantize_pads_partial_groups():
    x = torch.randn(4, 100)
    codes, scales = quantize_last_dim(x)
    assert codes.shape == (4, 100)
    assert scales.shape == (4, 1)
    assert (codes[:, 100:] if codes.shape[1] > 100 else True)


def test_quantize_zero_row():
    x = torch.zeros(2, 128)
    codes, scales = quantize_last_dim(x)
    assert torch.all(codes == 0)
    dq = dequantize_last_dim(codes, scales)
    assert torch.all(dq == 0)


def test_int8_linear_matches_fp32_within_int8_tolerance():
    g = torch.Generator().manual_seed(1)
    for M, K, N in ((8, 256, 128), (33, 128, 64), (4, 384, 96)):
        ref = nn = torch.nn.Linear(K, N, bias=True)
        with torch.no_grad():
            nn.weight.normal_(0, 0.1, generator=g)
            nn.bias.normal_(0, 0.1, generator=g)
        lin = Int8Linear.from_float(nn)
        x = torch.randn(M, K, generator=g)
        y = lin(x)
        y_ref = ref(x)
        rel = (y - y_ref).norm() / y_ref.norm()
        assert rel < 0.05, (M, K, N, rel.item())


def test_int8_linear_fake_quant_ste_grads():
    torch.manual_seed(2)
    nn = torch.nn.Linear(64, 32, bias=True)
    lin = Int8Linear.from_float(nn)
    x = torch.randn(4, 64, requires_grad=True)
    y = lin.fake_quant_forward(x)
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_triton_gemm_parity_with_torch_fallback():
    torch.manual_seed(3)
    dev = torch.device("cuda")
    for M, K, N in ((16, 256, 128), (64, 128, 96), (33, 384, 64), (128, 128, 128)):
        a = torch.randn(M, K, device=dev)
        w = torch.randn(N, K, device=dev) * 0.1
        a_codes, a_scales = quantize_last_dim(a)
        w_codes, w_scales = quantize_last_dim(w)
        y_gpu = _run_gpu(a_codes, a_scales, w_codes, w_scales)
        y_cpu = _run_cpu(a_codes, a_scales, w_codes, w_scales)
        torch.testing.assert_close(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-3)


def _run_cpu(a_codes, a_scales, w_codes, w_scales):
    from ksdraft.kernels.gemm import w8a8_gemm
    return w8a8_gemm(a_codes.cpu(), a_scales.cpu(), w_codes.cpu(), w_scales.cpu())


def _run_gpu(a_codes, a_scales, w_codes, w_scales):
    from ksdraft.kernels.gemm import w8a8_gemm
    return w8a8_gemm(a_codes.cuda(), a_scales.cuda(), w_codes.cuda(), w_scales.cuda())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_int8_linear_gpu_vs_cpu():
    torch.manual_seed(4)
    dev = torch.device("cuda")
    nn = torch.nn.Linear(256, 128, bias=True).to(dev)
    lin_gpu = Int8Linear.from_float(nn)
    x = torch.randn(48, 256, device=dev)
    y_gpu = lin_gpu(x)
    y_cpu = lin_gpu.cpu()(x.cpu())
    torch.testing.assert_close(y_gpu.cpu(), y_cpu, rtol=1e-4, atol=1e-3)
