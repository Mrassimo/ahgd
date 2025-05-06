"""
AHGD ETL Pipeline Package

This package provides ETL functionality for processing ABS geographic and census data.
"""

__version__ = '0.1.0'

# Basic utilities and core modules
from . import config
from . import utils
from . import geography
from . import census
from . import dimensions
from . import time_dimension
from . import validation

# Tables and table processing functions
from .tables import *

# Late import to avoid circular dependencies
"""
Handle attribute access for late import of modules to avoid circular dependencies.

Args:
   name (str): The name of the attribute being accessed.

Returns:
   The imported module if "run" is requested, otherwise raises AttributeError.
"""
def __getattr__(name):
   if name == 'run':
       from . import run
       return run
   raise AttributeError(f"module 'etl_logic' has no attribute '{name}'")

__all__ = [
    'config',
    'utils',
    'geography',
    'census',
    'dimensions',
    'time_dimension',
    'validation',
]