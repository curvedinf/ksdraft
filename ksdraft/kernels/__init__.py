"""INT8 tier: GS128 W8A8 kernels (RESEARCH.md Sections 8 and 12).

int8 is the intended dtype with the deepest optimization; bf16 is the generic
fallback tier implemented by the reference modules in ksdraft.model.
"""

from .quant import GROUP_SIZE, SCALE_DTYPE, quantize_last_dim, dequantize_last_dim
from .linear import Int8Linear

__all__ = [
    "GROUP_SIZE", "SCALE_DTYPE",
    "quantize_last_dim", "dequantize_last_dim",
    "Int8Linear",
]
