#!/usr/bin/env python3
"""
Quick check script for DSGE model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    from src.dsge_model import DSGEModel, ModelParameters
    print("✓ Successfully imported dsge_model")
except Exception as e:
    print(f"✗ Failed to import dsge_model: {e}")
    sys.exit(1)

# Test parameter loading
try:
    params = ModelParameters.from_json(os.path.join(os.path.dirname(__file__), 'config/parameters.json'))
    print("✓ Successfully loaded parameters")
    print(f"  Beta: {params.beta}")
    print(f"  Consumption tax: {params.tau_c}")
except Exception as e:
    print(f"✗ Failed to load parameters: {e}")
    sys.exit(1)

# Test model creation
try:
    model = DSGEModel(params)
    print("✓ Successfully created DSGE model")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    sys.exit(1)

# Test steady state
try:
    print("\nComputing steady state...")
    ss = model.compute_steady_state()
    print("✓ Successfully computed steady state")
    print(f"  GDP: {ss.Y:.3f}")
    print(f"  Consumption: {ss.C:.3f}")
    print(f"  Tax/GDP ratio: {ss.T/ss.Y:.1%}")
except Exception as e:
    print(f"✗ Failed to compute steady state: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All basic checks passed!")
