"""
Models subpackage for Japan Tax Simulator

This subpackage contains alternative model implementations:
- simple_dsge: Simplified DSGE model for rapid analysis
"""

from .simple_dsge import SimpleDSGEModel, SimpleDSGEParameters, SimpleSteadyState

__all__ = ['SimpleDSGEModel', 'SimpleDSGEParameters', 'SimpleSteadyState']