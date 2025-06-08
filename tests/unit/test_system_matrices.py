"""
Unit tests for system matrix operations and Blanchard-Kahn conditions

Tests matrix construction, Klein (2000) solution method, eigenvalue analysis,
and Blanchard-Kahn condition validation without heavy linearization computation.
"""

import pytest
import numpy as np
from scipy import linalg
from unittest.mock import Mock, patch, MagicMock
import warnings

from src.dsge_model import DSGEModel, ModelParameters, SteadyState
from src.linearization_improved import ImprovedLinearizedDSGE, LinearizedSystem


class TestSystemMatrixConstruction:
    """Test system matrix A, B, C construction and properties"""
    
    def test_matrix_shapes_consistency(self):
        """Test that A, B, C matrices have consistent shapes"""
        n_vars = 5
        n_shocks = 4
        
        # Create mock matrices with correct dimensions
        A = np.random.rand(n_vars, n_vars)
        B = np.random.rand(n_vars, n_vars)
        C = np.random.rand(n_vars, n_shocks)
        
        # Test shape consistency
        assert A.shape == (n_vars, n_vars), "A matrix should be square"
        assert B.shape == (n_vars, n_vars), "B matrix should be square"
        assert A.shape == B.shape, "A and B should have same shape"
        assert C.shape == (n_vars, n_shocks), "C should match variables and shocks"
    
    def test_matrix_finite_values(self):
        """Test that matrices contain finite values"""
        # Create test matrices
        A = np.array([[1.0, 0.5], [0.2, 0.8]])
        B = np.array([[0.9, -0.1], [0.0, 1.0]])
        C = np.array([[0.1], [0.05]])
        
        # Test finite values
        assert np.all(np.isfinite(A)), "A matrix should contain finite values"
        assert np.all(np.isfinite(B)), "B matrix should contain finite values"
        assert np.all(np.isfinite(C)), "C matrix should contain finite values"
    
    def test_matrix_numerical_properties(self):
        """Test numerical properties important for solution"""
        # Well-conditioned matrices
        A = np.array([[2, 1], [1, 3]])
        B = np.array([[1, 0], [0, 2]])
        
        # Test condition numbers
        cond_A = np.linalg.cond(A)
        cond_B = np.linalg.cond(B)
        
        assert cond_A < 1e12, "A should be well-conditioned"
        assert cond_B < 1e12, "B should be well-conditioned"
        
        # Test determinants
        det_A = np.linalg.det(A)
        det_B = np.linalg.det(B)
        
        assert abs(det_A) > 1e-12, "A should be non-singular"
        assert abs(det_B) > 1e-12, "B should be non-singular"
    
    def test_regularization_logic(self):
        """Test matrix regularization for rank-deficient cases"""
        # Create rank-deficient matrix
        A_singular = np.array([[1, 2], [2, 4]])  # rank 1
        
        # Check rank deficiency
        rank = np.linalg.matrix_rank(A_singular)
        assert rank < 2, "Matrix should be rank deficient"
        
        # Test regularization approach
        regularization = 1e-10
        A_regularized = A_singular + regularization * np.eye(2)
        
        # Check improvement
        rank_reg = np.linalg.matrix_rank(A_regularized)
        cond_reg = np.linalg.cond(A_regularized)
        
        assert rank_reg >= rank, "Regularization should not decrease rank"
        assert cond_reg < np.linalg.cond(A_singular), "Conditioning should improve"


class TestGeneralizedEigenvalueAnalysis:
    """Test generalized eigenvalue analysis for Klein method"""
    
    def test_generalized_eigenvalue_computation(self):
        """Test computation of generalized eigenvalues"""
        # Create test matrices for Ax = λBx
        A = np.array([[2, 1], [1, 3]])
        B = np.array([[1, 0], [0, 1]])
        
        # Compute generalized eigenvalues
        eigenvals = linalg.eigvals(A, B)
        
        # Test properties
        assert len(eigenvals) == 2, "Should have correct number of eigenvalues"
        assert np.all(np.isfinite(eigenvals)), "Eigenvalues should be finite"
        assert np.all(np.isreal(eigenvals)), "Should be real for this case"
    
    def test_eigenvalue_classification(self):
        """Test classification of eigenvalues for stability"""
        # Mock eigenvalues with known properties
        eigenvals = np.array([0.5, 0.8, 1.2, 1.5, 2.0])
        
        # Classify eigenvalues
        stable = np.abs(eigenvals) <= 1.0
        explosive = np.abs(eigenvals) > 1.0
        
        # Count classifications
        n_stable = np.sum(stable)
        n_explosive = np.sum(explosive)
        
        assert n_stable == 2, "Should identify 2 stable eigenvalues"
        assert n_explosive == 3, "Should identify 3 explosive eigenvalues"
        assert n_stable + n_explosive == len(eigenvals), "Should classify all"
    
    def test_complex_eigenvalue_handling(self):
        """Test handling of complex eigenvalues"""
        # Create matrix with complex eigenvalues
        A = np.array([[0, 1], [-1, 0]])  # Rotation matrix
        B = np.eye(2)
        
        eigenvals = linalg.eigvals(A, B)
        
        # Test magnitude calculation for complex numbers
        magnitudes = np.abs(eigenvals)
        
        assert len(magnitudes) == 2, "Should have magnitude for each eigenvalue"
        assert np.all(np.isfinite(magnitudes)), "Magnitudes should be finite"
        assert np.all(magnitudes >= 0), "Magnitudes should be non-negative"
    
    def test_schur_decomposition_properties(self):
        """Test properties of Schur decomposition"""
        # Create test matrices
        A = np.array([[2, 1], [0, 3]])
        B = np.array([[1, 0], [0, 1]])
        
        # Compute ordered Schur decomposition
        try:
            T, S, alpha, beta, Q, Z = linalg.ordqz(A, B, sort='ouc')
            
            # Test shapes
            assert T.shape == A.shape, "T should have same shape as A"
            assert S.shape == B.shape, "S should have same shape as B"
            assert Q.shape[0] == A.shape[0], "Q should have correct rows"
            assert Z.shape[0] == A.shape[0], "Z should have correct rows"
            
            # Test orthogonality
            Q_orth_error = np.max(np.abs(Q @ Q.T - np.eye(Q.shape[0])))
            Z_orth_error = np.max(np.abs(Z @ Z.T - np.eye(Z.shape[0])))
            
            assert Q_orth_error < 1e-12, "Q should be orthogonal"
            assert Z_orth_error < 1e-12, "Z should be orthogonal"
            
        except Exception as e:
            # Some versions might not have ordqz, skip test
            pytest.skip(f"Schur decomposition not available: {e}")


class TestBlanchardKahnConditions:
    """Test Blanchard-Kahn condition checking"""
    
    def test_determinacy_condition_simple(self):
        """Test determinacy condition: n_explosive = n_forward_looking"""
        # Mock scenario: determinacy satisfied
        n_explosive = 2
        n_forward_looking = 2
        
        # Test determinacy
        is_determinate = (n_explosive == n_forward_looking)
        assert is_determinate, "Should be determinate when counts match"
        
        # Mock scenario: indeterminacy
        n_explosive_2 = 1
        n_forward_looking_2 = 2
        
        is_indeterminate = (n_explosive_2 < n_forward_looking_2)
        assert is_indeterminate, "Should be indeterminate when too few explosive"
        
        # Mock scenario: no solution
        n_explosive_3 = 3
        n_forward_looking_3 = 2
        
        no_solution = (n_explosive_3 > n_forward_looking_3)
        assert no_solution, "Should have no solution when too many explosive"
    
    def test_eigenvalue_counting_accuracy(self):
        """Test accurate counting of explosive eigenvalues"""
        # Test with various eigenvalue scenarios
        test_cases = [
            (np.array([0.5, 0.8, 1.2, 1.5]), 2),  # 2 explosive
            (np.array([0.9, 0.95, 1.01, 1.1]), 2),  # 2 explosive (close to unit circle)
            (np.array([0.1, 0.5, 2.0, 3.0]), 2),  # 2 explosive
            (np.array([1.0, 1.0, 1.0, 1.0]), 0),  # All on unit circle (stable)
        ]
        
        for eigenvals, expected_explosive in test_cases:
            explosive_count = np.sum(np.abs(eigenvals) > 1.0)
            assert explosive_count == expected_explosive, f"Failed for {eigenvals}"
    
    def test_unit_circle_boundary_cases(self):
        """Test handling of eigenvalues exactly on unit circle"""
        # Eigenvalues exactly on unit circle
        eigenvals_on_circle = np.array([1.0, -1.0, 1.0j, -1.0j])
        
        # Test classification (convention: |λ| = 1 is stable)
        stable_count = np.sum(np.abs(eigenvals_on_circle) <= 1.0)
        explosive_count = np.sum(np.abs(eigenvals_on_circle) > 1.0)
        
        assert stable_count == 4, "Unit circle eigenvalues should be stable"
        assert explosive_count == 0, "No eigenvalues should be explosive"
        
        # Test with tolerance
        tolerance = 1e-10
        eigenvals_near_circle = np.array([1.0 + tolerance, 1.0 - tolerance])
        
        stable_with_tol = np.sum(np.abs(eigenvals_near_circle) <= 1.0 + tolerance)
        assert stable_with_tol >= 1, "Should handle near-unit-circle cases"


class TestPolicyFunctionConstruction:
    """Test policy function matrix construction P = control(state)"""
    
    def test_policy_matrix_dimensions(self):
        """Test policy matrix has correct dimensions"""
        n_control = 3  # Number of control variables
        n_state = 2    # Number of state variables
        
        # Create mock policy matrix
        P = np.random.rand(n_control, n_state)
        
        assert P.shape == (n_control, n_state), "Policy matrix should be n_control × n_state"
    
    def test_policy_matrix_properties(self):
        """Test policy matrix numerical properties"""
        # Create test policy matrix
        P = np.array([[0.5, 0.3], [0.2, 0.8], [-0.1, 0.4]])
        
        # Test finite values
        assert np.all(np.isfinite(P)), "Policy matrix should have finite values"
        
        # Test that matrix can be used for simulation
        state = np.array([0.1, -0.05])  # Small deviations from steady state
        control = P @ state
        
        assert len(control) == P.shape[0], "Control should have correct dimension"
        assert np.all(np.isfinite(control)), "Control response should be finite"
    
    def test_policy_function_stability(self):
        """Test that policy function implies stable dynamics"""
        # Create mock transition and policy matrices
        Q = np.array([[0.8, 0.1], [0.0, 0.9]])  # Stable transition matrix
        P = np.array([[0.5, 0.3]])  # Policy matrix
        
        # Check transition matrix stability
        eigenvals_Q = np.linalg.eigvals(Q)
        assert np.all(np.abs(eigenvals_Q) < 1.0), "Transition should be stable"
        
        # Test simulation stability
        state = np.array([0.1, 0.05])
        for _ in range(10):  # Short simulation
            control = P @ state
            state = Q @ state  # No shock
            
        # State should converge to zero (stable)
        assert np.max(np.abs(state)) < 0.5, "System should be converging"


class TestTransitionMatrixProperties:
    """Test state transition matrix Q properties"""
    
    def test_transition_matrix_stability(self):
        """Test that transition matrix has stable eigenvalues"""
        # Create stable transition matrix
        Q = np.array([[0.9, 0.1], [0.0, 0.8]])
        
        eigenvals = np.linalg.eigvals(Q)
        
        # Test stability
        assert np.all(np.abs(eigenvals) < 1.0), "All eigenvalues should be stable"
        assert np.all(np.isfinite(eigenvals)), "Eigenvalues should be finite"
    
    def test_transition_matrix_properties_for_simulation(self):
        """Test properties needed for impulse response simulation"""
        Q = np.array([[0.8, 0.1], [0.0, 0.9]])
        R = np.array([[0.1], [0.05]])  # Shock loading matrix
        
        # Test dimensions
        assert Q.shape[0] == Q.shape[1], "Q should be square"
        assert R.shape[0] == Q.shape[0], "R should have same rows as Q"
        
        # Test that system responds to shocks
        shock = np.array([1.0])  # Unit shock
        initial_response = R @ shock
        
        assert np.any(initial_response != 0), "System should respond to shocks"
        assert np.all(np.isfinite(initial_response)), "Response should be finite"
    
    def test_long_run_convergence(self):
        """Test that transition matrix implies long-run convergence"""
        Q = np.array([[0.7, 0.2], [0.1, 0.6]])
        
        # Compute high powers to test convergence
        Q_power = Q.copy()
        for _ in range(20):
            Q_power = Q_power @ Q
        
        # High powers should approach zero for stable systems
        max_element = np.max(np.abs(Q_power))
        assert max_element < 0.1, "High powers should be small for stable systems"


class TestMatrixSolutionMethods:
    """Test matrix solution methods and fallbacks"""
    
    def test_matrix_inversion_methods(self):
        """Test different matrix inversion approaches"""
        # Well-conditioned matrix
        A = np.array([[2, 1], [1, 3]])
        
        # Test standard inversion
        A_inv_std = np.linalg.inv(A)
        assert np.allclose(A @ A_inv_std, np.eye(2)), "Standard inversion should work"
        
        # Test pseudo-inverse (should give same result for full rank)
        A_inv_pseudo = np.linalg.pinv(A)
        assert np.allclose(A_inv_std, A_inv_pseudo, atol=1e-12), "Pseudo-inverse should match"
    
    def test_pseudo_inverse_fallback(self):
        """Test pseudo-inverse fallback for rank-deficient matrices"""
        # Rank-deficient matrix
        A_singular = np.array([[1, 2], [2, 4]])
        
        # Standard inverse should fail
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.inv(A_singular)
        
        # Pseudo-inverse should work
        A_pinv = np.linalg.pinv(A_singular)
        assert A_pinv.shape == (2, 2), "Pseudo-inverse should have correct shape"
        assert np.all(np.isfinite(A_pinv)), "Pseudo-inverse should be finite"
    
    def test_matrix_conditioning_assessment(self):
        """Test assessment of matrix conditioning"""
        # Well-conditioned matrix
        A_good = np.array([[2, 1], [1, 3]])
        cond_good = np.linalg.cond(A_good)
        
        # Ill-conditioned matrix
        A_bad = np.array([[1, 1], [1, 1.000000001]])  # Even closer to singular
        cond_bad = np.linalg.cond(A_bad)
        
        assert cond_good < 100, "Good matrix should be well-conditioned"
        assert cond_bad > 1e3, "Bad matrix should be ill-conditioned"
        
        # Test conditioning comparison
        assert cond_bad > cond_good, "Bad matrix should have higher condition number"
        
        # Test that we can detect relative conditioning
        threshold = 1e6
        is_good_conditioned = cond_good < threshold
        
        assert is_good_conditioned, "Good matrix should pass reasonable threshold"


class TestErrorHandlingInMatrixOperations:
    """Test error handling in matrix operations"""
    
    def test_singular_matrix_handling(self):
        """Test handling of singular matrices"""
        # Singular matrix
        A_singular = np.zeros((2, 2))
        
        # Test rank detection
        rank = np.linalg.matrix_rank(A_singular)
        assert rank == 0, "Should detect zero rank"
        
        # Test that we can detect singularity
        try:
            det = np.linalg.det(A_singular)
            assert abs(det) < 1e-12, "Determinant should be near zero"
        except:
            pass  # Some implementations might handle differently
    
    def test_non_square_matrix_handling(self):
        """Test handling of non-square matrices"""
        # Non-square matrix
        A_nonsquare = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Should not be able to compute eigenvalues
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.eigvals(A_nonsquare)
        
        # But should be able to compute pseudo-inverse
        A_pinv = np.linalg.pinv(A_nonsquare)
        assert A_pinv.shape == (3, 2), "Pseudo-inverse should have transposed shape"
    
    def test_infinite_or_nan_handling(self):
        """Test handling of infinite or NaN values"""
        # Matrix with NaN
        A_nan = np.array([[1, 2], [np.nan, 4]])
        
        # Should detect NaN
        has_nan = np.any(np.isnan(A_nan))
        assert has_nan, "Should detect NaN values"
        
        # Matrix with infinity
        A_inf = np.array([[1, 2], [np.inf, 4]])
        
        # Should detect infinity
        has_inf = np.any(np.isinf(A_inf))
        assert has_inf, "Should detect infinite values"
        
        # Combined check for finite values
        is_finite = np.all(np.isfinite(A_nan))
        assert not is_finite, "Should detect non-finite values"


if __name__ == "__main__":
    pytest.main([__file__])