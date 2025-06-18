"""
Australian Health Analytics Dashboard Package

This package contains the modular components of the health analytics dashboard,
organised into clean, testable modules.

Modules:
    data: Data loading and processing functionality
"""

__version__ = "1.0.0"
__author__ = "Australian Health Analytics Team"

# Package-level imports for convenience
from .data import loaders, processors

__all__ = ['loaders', 'processors']