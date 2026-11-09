"""ksdraft — parallel block-drafting draft models (RWKV-8 Rosa + KS3).

Design source of truth: RESEARCH.md in the repository root.
"""

__version__ = "0.1.0"

from .config import KSDraftConfig, Dtype

__all__ = ["KSDraftConfig", "Dtype", "__version__"]
