"""
Japan Tax Simulator - DSGE Model Package

This package provides a comprehensive Dynamic Stochastic General Equilibrium (DSGE)
model for analyzing the impact of tax policy changes on the Japanese economy.
"""

from .dsge_model import DSGEModel, ModelParameters, SteadyState

__all__ = [
    'DSGEModel',
    'ModelParameters', 
    'SteadyState',
]

__version__ = '1.0.0'
