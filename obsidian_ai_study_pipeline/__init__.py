"""
Obsidian AI Study Pipeline

A comprehensive tool for generating AI-powered quiz questions and answers from Obsidian vaults.
This package provides functionality for parsing vault content, semantic search, and quiz generation.
"""

__version__ = "0.1.0"
__author__ = "Obsidian AI Study Pipeline Contributors"

# Make imports optional to allow testing without dependencies
try:
    from .pipeline import ObsidianStudyPipeline
    __all__ = ["ObsidianStudyPipeline"]
except ImportError as e:
    import warnings
    warnings.warn(f"Some dependencies not available: {e}. Install with: pip install -r requirements.txt")
    __all__ = []