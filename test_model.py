#!/usr/bin/env python3
"""
Test script for DSGE model implementation
Tests basic functionality and validates model consistency
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dsge_model import DSGEModel, ModelParameters, load_model
from src.linearization_improved import ImprovedLinearizedDSGE
from src.tax_simulator import EnhancedTaxSimulator, TaxReform


def test_steady_state_computation():
    """Test steady state computation and validation"""
    print("=" * 60)
    print("Testing Steady State Computation")
    print("=" * 60)
    
    # Load model
    config_path = os.path.join(os.path.dirname(__file__), 'config/parameters.json')
    model = load_model(config_path)
    
    # Compute steady state
    try:
        steady_state = model.compute_steady_state()
        print("✓ Steady state computation successful")
        
        # Check steady state conditions
        errors = model.check_steady_state(steady_state)
        
        print("\nSteady State Validation:")
        for condition, error in errors.items():
            status = "✓" if abs(error) < 1e-6 else "✗"
            print(f"  {status} {condition}: {error:.6f}")
        
        # Display key steady state values
        print("\nKey Steady State Values:")
        print(f"  GDP (Y): {steady_state.Y:.3f}")
        print(f"  Consumption (C): {steady_state.C:.3f}")
        print(f"  Investment (I): {steady_state.I:.3f}")
        print(f"  Labor (L): {steady_state.L:.3f}")
        print(f"  Tax Revenue (T): {steady_state.T:.3f}")
        print(f"  Tax/GDP ratio: {steady_state.T/steady_state.Y:.1%}")
        
        return True, model, steady_state
        
    except Exception as e:
        print(f"✗ Steady state computation failed: {e}")
        return False, None, None


def test_linearization():
    """Test model linearization using Klein method"""
    print("\n" + "=" * 60)
    print("Testing Model Linearization")
    print("=" * 60)
    
    success, model, steady_state = test_steady_state_computation()
    if not success:
        return False
    
    try:
        # Create linearized model
        linear_model = ImprovedLinearizedDSGE(model, steady_state)
        
        # Build system matrices
        system = linear_model.build_system_matrices()
        print(f"✓ System matrices built: A={system.A.shape}, B={system.B.shape}")
        
        # Solve using Klein method
        P, Q = linear_model.solve_klein()
        print(f"✓ Klein solution successful: P={P.shape}, Q={Q.shape}")
        
        # Check eigenvalues
        eigenvalues = np.linalg.eigvals(Q)
        n_stable = np.sum(np.abs(eigenvalues) < 1.0)
        n_explosive = np.sum(np.abs(eigenvalues) > 1.0)
        
        print(f"\nEigenvalue Analysis:")
        print(f"  Stable eigenvalues: {n_stable}")
        print(f"  Explosive eigenvalues: {n_explosive}")
        print(f"  Max eigenvalue: {np.max(np.abs(eigenvalues)):.3f}")
        
        return True, linear_model
        
    except Exception as e:
        print(f"✗ Linearization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_impulse_responses():
    """Test impulse response functions"""
    print("\n" + "=" * 60)
    print("Testing Impulse Response Functions")
    print("=" * 60)
    
    success, linear_model = test_linearization()
    if not success:
        return False
    
    shock_types = ['tfp', 'gov_spending', 'monetary', 'consumption_tax']
    
    for shock in shock_types:
        try:
            # Compute impulse response
            if 'tax' in shock:
                irf = linear_model.compute_impulse_response(shock, shock_size=1.0, periods=20)
            else:
                irf = linear_model.compute_impulse_response(shock, shock_size=1.0, periods=20)
            
            # Check key responses
            gdp_impact = irf['Y'].iloc[0]
            gdp_peak = irf['Y'].abs().max()
            
            print(f"\n{shock.replace('_', ' ').title()} Shock:")
            print(f"  ✓ IRF computed successfully")
            print(f"  GDP impact: {gdp_impact:+.3f}%")
            print(f"  GDP peak response: {gdp_peak:.3f}%")
            
        except Exception as e:
            print(f"  ✗ IRF computation failed for {shock}: {e}")
            return False
    
    return True


def test_tax_simulation():
    """Test tax reform simulation"""
    print("\n" + "=" * 60)
    print("Testing Tax Reform Simulation")
    print("=" * 60)
    
    # Load baseline model
    config_path = os.path.join(os.path.dirname(__file__), 'config/parameters.json')
    baseline_model = load_model(config_path)
    baseline_model.compute_steady_state()
    
    # Create tax simulator
    try:
        tax_simulator = EnhancedTaxSimulator(baseline_model)
        print("✓ Tax simulator initialized")
        
        # Test simple consumption tax reform
        reform = TaxReform(
            name="Test Reform",
            tau_c=0.15,  # 10% to 15%
            implementation='permanent'
        )
        
        results = tax_simulator.simulate_reform(reform, periods=50, compute_welfare=True)
        
        print(f"\nConsumption Tax Reform (10% → 15%):")
        print(f"  ✓ Simulation completed")
        print(f"  Welfare change: {results.welfare_change:+.3f}%")
        print(f"  Long-run GDP effect: {(results.steady_state_reform.Y - results.steady_state_baseline.Y) / results.steady_state_baseline.Y * 100:+.3f}%")
        print(f"  Long-run tax revenue effect: {(results.steady_state_reform.T - results.steady_state_baseline.T) / results.steady_state_baseline.T * 100:+.3f}%")
        print(f"  Transition period: {results.transition_periods} quarters")
        
        return True
        
    except Exception as e:
        print(f"✗ Tax simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_consistency():
    """Test model internal consistency"""
    print("\n" + "=" * 60)
    print("Testing Model Consistency")
    print("=" * 60)
    
    config_path = os.path.join(os.path.dirname(__file__), 'config/parameters.json')
    model = load_model(config_path)
    params = model.params
    
    # Test parameter bounds
    tests = [
        ("Discount factor", 0 < params.beta < 1),
        ("Capital share", 0 < params.alpha < 1),
        ("Depreciation rate", 0 < params.delta < 1),
        ("Price stickiness", 0 < params.theta_p < 1),
        ("Tax rates", all(0 <= rate <= 1 for rate in [params.tau_c, params.tau_l, params.tau_k, params.tau_f])),
        ("Elasticities", all(p > 0 for p in [params.sigma_c, params.sigma_l, params.epsilon])),
    ]
    
    all_passed = True
    for test_name, condition in tests:
        status = "✓" if condition else "✗"
        print(f"  {status} {test_name}")
        if not condition:
            all_passed = False
    
    return all_passed


def main():
    """Run all tests"""
    print("\nDSGE Model Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Consistency", test_model_consistency),
        ("Steady State", lambda: test_steady_state_computation()[0]),
        ("Linearization", lambda: test_linearization()[0]),
        ("Impulse Responses", test_impulse_responses),
        ("Tax Simulation", test_tax_simulation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    n_passed = sum(1 for _, success in results if success)
    n_total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {test_name}: {status}")
    
    print(f"\nTotal: {n_passed}/{n_total} tests passed")
    
    return n_passed == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
