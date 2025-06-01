"""Test linearization and impulse response functions"""

from src.dsge_model import DSGEModel, ModelParameters
from src.linearization_improved import ImprovedLinearizedDSGE
import numpy as np

# Load model and compute steady state
params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)
print("Computing steady state...")
ss = model.compute_steady_state()
print(f"✓ Steady state computed: Y={ss.Y:.3f}, C={ss.C:.3f}")

# Test linearization
try:
    print("\nTesting linearization...")
    linearizer = ImprovedLinearizedDSGE(model, ss)
    
    # Get model equations
    equations = model.get_model_equations()
    print(f"✓ Retrieved {len(equations)} model equations")
    
    # Test system matrix construction
    print("Building system matrices...")
    try:
        linear_system = linearizer.build_system_matrices()
        print(f"✓ System matrices built:")
        print(f"  A shape: {linear_system.A.shape}")
        print(f"  B shape: {linear_system.B.shape}")
        print(f"  C shape: {linear_system.C.shape}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(linear_system.A)) or np.any(np.isinf(linear_system.A)):
            print("⚠ Warning: A contains NaN or infinite values")
        if np.any(np.isnan(linear_system.B)) or np.any(np.isinf(linear_system.B)):
            print("⚠ Warning: B contains NaN or infinite values")
            
        # Check condition numbers
        cond_A = np.linalg.cond(linear_system.A)
        print(f"  A condition number: {cond_A:.2e}")
        if cond_A > 1e12:
            print("⚠ Warning: A is ill-conditioned")
            
    except Exception as e:
        print(f"✗ System matrix construction failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
        
    # Test Klein solution
    print("\nTesting Klein solution...")
    try:
        P, Q = linearizer.solve_klein()
        print(f"✓ Klein solution succeeded")
        print(f"  P shape: {P.shape}, Q shape: {Q.shape}")
        
        # Check if P and Q contain reasonable values
        if np.any(np.isnan(P)) or np.any(np.isinf(P)):
            print("⚠ Warning: P contains NaN or infinite values")
        if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
            print("⚠ Warning: Q contains NaN or infinite values")
            
        max_P = np.max(np.abs(P))
        max_Q = np.max(np.abs(Q))
        print(f"  Max |P|: {max_P:.6f}, Max |Q|: {max_Q:.6f}")
        
        # Store for impulse response test
        klein_success = True
            
    except Exception as e:
        print(f"✗ Klein solution error: {e}")
        import traceback
        traceback.print_exc()
        klein_success = False
        
    # Test impulse response computation
    if klein_success:
        print("\nTesting impulse response computation...")
        try:
            # Test a TFP shock
            shock_size = 1.0  # 1% shock
            periods = 20
            
            # Compute impulse response
            irf_result = linearizer.compute_impulse_response('tfp', shock_size, periods)
            
            if irf_result is not None and not irf_result.empty:
                print(f"✓ Impulse response computed")
                print(f"  Periods: {len(irf_result)}")
                print(f"  Variables: {list(irf_result.columns)}")
                
                # Check if responses are non-zero
                max_response = np.max(np.abs(irf_result.values))
                print(f"  Max absolute response: {max_response:.6e}")
                
                if max_response < 1e-10:
                    print("⚠ WARNING: All impulse responses are essentially zero!")
                else:
                    # Show some key responses
                    if 'Y' in irf_result.columns:
                        y_response = irf_result['Y'].values
                        max_y_response = np.max(np.abs(y_response))
                        print(f"  Max Y response: {max_y_response:.6e}")
                    
                    if 'C' in irf_result.columns:
                        c_response = irf_result['C'].values
                        max_c_response = np.max(np.abs(c_response))
                        print(f"  Max C response: {max_c_response:.6e}")
                        
            else:
                print("✗ Impulse response computation returned None or empty")
                
        except Exception as e:
            print(f"✗ Impulse response computation failed: {e}")
            import traceback
            traceback.print_exc()
    
except Exception as e:
    print(f"✗ Linearization test failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDynamic testing completed.")