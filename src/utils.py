"""
Utility functions for safe mathematical operations
"""
import numpy as np


def safe_divide(numerator, denominator, default=0.0, tolerance=1e-10):
    """
    Safe division that handles zero denominators
    
    Args:
        numerator: The numerator value
        denominator: The denominator value
        default: Default value to return if division by zero
        tolerance: Minimum threshold to consider as zero
    
    Returns:
        Result of division or default value
    """
    if abs(denominator) < tolerance:
        return default
    return numerator / denominator


def safe_percentage_change(new_value, old_value, default=0.0, tolerance=1e-10):
    """
    Safe percentage change calculation that handles zero base values
    
    Args:
        new_value: New value
        old_value: Original value 
        default: Default percentage change when base is zero
        tolerance: Minimum threshold to consider as zero
    
    Returns:
        Percentage change or default value
    """
    if abs(old_value) < tolerance:
        if abs(new_value) < tolerance:
            return 0.0  # Both are zero, no change
        else:
            return default  # Base is zero but new value isn't
    
    return (new_value - old_value) / old_value * 100


def validate_economic_variables(variables_dict, min_values=None):
    """
    Validate that economic variables are within reasonable bounds
    
    Args:
        variables_dict: Dictionary of variable names and values
        min_values: Dictionary of minimum acceptable values
    
    Returns:
        Tuple of (is_valid, warnings)
    """
    if min_values is None:
        min_values = {
            'Y': 1e-6,  # GDP should be positive
            'C': 1e-6,  # Consumption should be positive
            'L': 1e-6,  # Labor should be positive
            'K': 1e-6,  # Capital should be positive
            'I': -1.0,  # Investment can be negative (disinvestment)
        }
    
    warnings = []
    is_valid = True
    
    for var_name, value in variables_dict.items():
        if var_name in min_values:
            min_val = min_values[var_name]
            if value < min_val:
                warnings.append(f"{var_name} = {value:.6f} is below minimum {min_val}")
                is_valid = False
    
    return is_valid, warnings


def robust_ratio_calculation(numerator, denominator, ratio_name="ratio", 
                           max_ratio=1000, min_denominator=1e-10):
    """
    Robust calculation of economic ratios with bounds checking
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        ratio_name: Name of the ratio for error messages
        max_ratio: Maximum acceptable ratio value
        min_denominator: Minimum denominator to avoid division by zero
    
    Returns:
        Calculated ratio with bounds checking
    """
    if abs(denominator) < min_denominator:
        print(f"Warning: {ratio_name} denominator too small ({denominator:.2e}), using minimum")
        denominator = min_denominator if denominator >= 0 else -min_denominator
    
    ratio = numerator / denominator
    
    if abs(ratio) > max_ratio:
        print(f"Warning: {ratio_name} = {ratio:.2f} exceeds maximum {max_ratio}")
        ratio = max_ratio if ratio > 0 else -max_ratio
    
    return ratio