"""Model components (reference tier: fp32/bf16, pure torch, autograd).

The backbone is RWKV-8 proper: the Rosa hard-retrieval mixer is the
time-mixing layer, with KS3 spine decoders. The optimized INT8 tier
(W8A8 GS128 fused kernels) lives in ksdraft.kernels.
"""

from .ks3 import KS3, spine_horner, spine_basis
from .rosa import RosaMixer
from .spine import AdjacentInjection
from .block import DraftBlock
from .draft import KSDraftModel

__all__ = [
    "KS3", "spine_horner", "spine_basis", "RosaMixer",
    "AdjacentInjection", "DraftBlock", "KSDraftModel",
]
