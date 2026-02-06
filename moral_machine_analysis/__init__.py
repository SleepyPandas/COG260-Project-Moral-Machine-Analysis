"""Core analysis package for the Moral Machine dataset."""

from .config import RunConfig
from .pipeline import run_analysis

__all__ = ["RunConfig", "run_analysis"]
