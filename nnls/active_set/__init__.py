"""
Module for collection of active set methods for solving nonnegative least squares.
"""

# Authors: Joseph Knox <joseph.edward.knox@gmail.com>
# License: BSD 3

from .lawson_hanson import lawson_hanson


__all__ = ['lawson_hanson']
