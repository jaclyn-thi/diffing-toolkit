"""
Diffing methods for comparing models.
"""

from .kl import KLDivergenceDiffingMethod
from .activation_analysis import ActivationAnalysisDiffingMethod

__all__ = [
    "KLDivergenceDiffingMethod",
    "ActivationAnalysisDiffingMethod",
]

try:
    from .crosscoder import CrosscoderDiffingMethod
except ImportError:
    CrosscoderDiffingMethod = None  # type: ignore[misc, assignment]
else:
    __all__.append("CrosscoderDiffingMethod")

try:
    from .sae_difference import SAEDifferenceMethod
except ImportError:
    SAEDifferenceMethod = None  # type: ignore[misc, assignment]
else:
    __all__.append("SAEDifferenceMethod")
