"""
brain/__init__.py
=================
Public API surface for the Brain module.

The only things an external caller needs:

    from brain import Brain, AnalysisResult
    from brain import train_all

Everything else (rule engine internals, feature extractor, model loaders)
is an implementation detail and should not be imported directly by the API
layer or UI layer.
"""

from brain.pipeline import Brain, AnalysisResult
from brain.training import train_all

__all__ = ["Brain", "AnalysisResult", "train_all"]
