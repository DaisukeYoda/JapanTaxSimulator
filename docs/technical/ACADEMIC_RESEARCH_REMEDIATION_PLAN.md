# Academic Research Remediation Plan
## Japan Tax Simulator - Comprehensive Code Overhaul

**Date Created:** 2025-06-08  
**Priority:** CRITICAL - Research Integrity at Risk  
**Estimated Timeline:** 4-6 weeks for complete remediation

---

## 🚨 IMMEDIATE CRITICAL ACTIONS (Week 1)

### 1. Code Quarantine and Warning System
- [ ] Add `@RESEARCH_WARNING` decorators to all functions with dummy values
- [ ] Implement mandatory warnings for fallback usage
- [ ] Create research mode vs. development mode flags
- [ ] Add explicit error logging for all silent failures

```python
# IMMEDIATE IMPLEMENTATION
@research_warning("Uses fallback linearization - results may be unreliable")
def _setup_simple_linearization(self):
    raise NotImplementedError(
        "Simplified linearization not validated for research use. "
        "Please resolve underlying Blanchard-Kahn condition violations."
    )
```

### 2. Disable Dangerous Fallbacks
- [ ] Comment out all `DummySteadyState` usage
- [ ] Remove automatic model switching (simple ↔ complex)
- [ ] Disable automatic regularization of singular matrices
- [ ] Force explicit convergence verification

### 3. Parameter Source Documentation
- [ ] Audit `config/parameters.json` for empirical citations
- [ ] Create `PARAMETER_SOURCES.md` with academic references
- [ ] Flag all uncited parameters as "REQUIRES_EMPIRICAL_VALIDATION"

---

## 📊 PHASE 1: DATA INTEGRITY (Weeks 1-2)

### A. Parameter Verification and Documentation

**File:** `src/parameter_validation.py` (NEW)
```python
class EmpiricalParameter:
    def __init__(self, value: float, source: str, year: int, confidence_interval: tuple):
        self.value = value
        self.source = source  # e.g., "Miyazaki & Yoda (2019), Table 3"
        self.year = year
        self.ci_lower, self.ci_upper = confidence_interval
    
    def validate_usage(self, context: str) -> None:
        if self.year < 2010:
            warnings.warn(f"Parameter from {self.year} may be outdated for {context}")
```

**Required Sources:**
- Labor supply elasticity: Keane & Rogerson (2012), Japanese Labor Force Survey
- Consumption elasticity: Ogaki & Reinhart (1998), Japanese household data
- Capital share: Miyazaki (2019), Japanese national accounts
- Tax elasticities: Hamada et al. (2020), MOF empirical analysis

### B. Empirical Data Integration

**File:** `src/empirical_data.py` (NEW)
```python
class JapaneseEmpiricalData:
    """
    Empirical data manager for Japanese economy
    All data must have verifiable sources and be within valid date ranges
    """
    
    @staticmethod
    def get_tax_composition(year: int) -> Dict[str, float]:
        """
        Source: Ministry of Finance, Annual Tax Statistics
        Valid years: 2000-2023
        """
        if year < 2000 or year > 2023:
            raise ValueError(f"Tax data only available for 2000-2023, got {year}")
        
        # Implementation required: Connect to MOF data
        raise NotImplementedError("Must implement MOF data connection")
    
    @staticmethod
    def get_macro_ratios(year: int) -> Dict[str, float]:
        """
        Source: Cabinet Office, National Accounts of Japan
        Returns: C/Y, I/Y, G/Y ratios with uncertainty bounds
        """
        raise NotImplementedError("Must implement Cabinet Office data connection")
```

---

## 🔧 PHASE 2: COMPUTATIONAL RIGOR (Weeks 2-3)

### A. Convergence and Stability Framework

**File:** `src/numerical_stability.py` (NEW)
```python
class ConvergenceValidator:
    """Strict convergence checking for research applications"""
    
    @staticmethod
    def verify_steady_state(equations_residual: np.ndarray, 
                          tolerance: float = 1e-10) -> None:
        """
        Strict steady state verification - no relaxation allowed
        
        Args:
            equations_residual: Residuals from steady state equations
            tolerance: Absolute tolerance (default: 1e-10)
            
        Raises:
            ConvergenceError: If any residual exceeds tolerance
        """
        max_residual = np.max(np.abs(equations_residual))
        if max_residual > tolerance:
            raise ConvergenceError(
                f"Steady state failed convergence test. "
                f"Maximum residual: {max_residual:.2e} > {tolerance:.2e}. "
                f"Problematic equations: {np.where(np.abs(equations_residual) > tolerance)[0]}"
            )
    
    @staticmethod
    def verify_blanchard_kahn(eigenvalues: np.ndarray, 
                            n_predetermined: int) -> None:
        """
        Strict Blanchard-Kahn condition verification
        
        Raises:
            BKConditionError: If conditions not satisfied
        """
        n_explosive = np.sum(np.abs(eigenvalues) > 1.0 + 1e-10)
        n_jump = len(eigenvalues) - n_predetermined
        
        if n_explosive != n_jump:
            raise BKConditionError(
                f"Blanchard-Kahn conditions violated. "
                f"Need {n_jump} explosive eigenvalues, found {n_explosive}. "
                f"Model solution is not unique/stable."
            )

class ResearchModelSolver:
    """Research-grade model solver with strict validation"""
    
    def solve_steady_state(self, initial_guess: np.ndarray, 
                          max_iterations: int = 1000,
                          tolerance: float = 1e-10) -> SteadyState:
        """
        Solve steady state with research-grade requirements
        
        Returns: SteadyState object with validation metadata
        Raises: Various specific exceptions for different failure modes
        """
        # Single solver method - no fallbacks
        result = optimize.root(
            self.steady_state_equations, 
            initial_guess,
            method='hybr',  # Specific, well-documented method
            options={'xtol': tolerance, 'maxfev': max_iterations}
        )
        
        if not result.success:
            raise SteadyStateError(
                f"Steady state solver failed: {result.message}. "
                f"Final function norm: {np.linalg.norm(result.fun):.2e}. "
                f"Iterations: {result.nfev}/{max_iterations}. "
                f"Consider: 1) Parameter bounds, 2) Initial guess, 3) Model specification"
            )
        
        # Strict validation
        ConvergenceValidator.verify_steady_state(result.fun, tolerance)
        
        return self._create_validated_steady_state(result.x)
```

### B. Linearization Overhaul

**File:** `src/research_linearization.py` (NEW)
```python
class ResearchLinearization:
    """Research-grade linearization without fallbacks or regularization"""
    
    def __init__(self, model: DSGEModel, steady_state: SteadyState):
        self.model = model
        self.steady_state = steady_state
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate model and steady state before linearization"""
        # Check steady state residuals
        residuals = self.model.evaluate_equations(self.steady_state)
        if np.max(np.abs(residuals)) > 1e-10:
            raise ValidationError("Steady state not sufficiently accurate for linearization")
        
        # Check economic ratios
        if not (0.4 <= self.steady_state.C/self.steady_state.Y <= 0.8):
            raise ValidationError(f"Consumption ratio {self.steady_state.C/self.steady_state.Y:.3f} outside realistic bounds")
    
    def linearize(self) -> LinearizedSystem:
        """
        Perform linearization with strict validation
        
        Returns: LinearizedSystem with full diagnostic information
        Raises: LinearizationError if system cannot be linearized properly
        """
        # Build system matrices using symbolic differentiation
        A, B, C = self._build_system_matrices()
        
        # Validate matrix properties
        self._validate_system_matrices(A, B, C)
        
        # Solve using Klein method (no fallbacks)
        P, Q = self._solve_klein_method(A, B)
        
        return LinearizedSystem(A=A, B=B, C=C, P=P, Q=Q, 
                              diagnostic_info=self._get_diagnostics())
    
    def _validate_system_matrices(self, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
        """Validate system matrices without any modifications"""
        # Check ranks
        rank_A = np.linalg.matrix_rank(A)
        rank_B = np.linalg.matrix_rank(B)
        
        if rank_A < A.shape[0]:
            raise LinearizationError(f"A matrix is rank deficient: {rank_A}/{A.shape[0]}")
        
        if rank_B < B.shape[0]:
            raise LinearizationError(f"B matrix is rank deficient: {rank_B}/{B.shape[0]}")
        
        # No regularization - fail if matrices are problematic
        condition_A = np.linalg.cond(A)
        if condition_A > 1e12:
            raise LinearizationError(f"A matrix is ill-conditioned: condition number {condition_A:.2e}")
```

---

## 📈 PHASE 3: RESEARCH ANALYSIS FRAMEWORK (Weeks 3-4)

### A. Tax Reform Analysis with Uncertainty

**File:** `src/research_tax_analysis.py` (NEW)
```python
class ResearchTaxSimulator:
    """Research-grade tax simulator with uncertainty quantification"""
    
    def __init__(self, model: DSGEModel, parameter_uncertainty: Dict[str, float]):
        self.model = model
        self.parameter_uncertainty = parameter_uncertainty
        self._validate_model()
    
    def simulate_reform_with_uncertainty(self, reform: TaxReform, 
                                       n_monte_carlo: int = 1000) -> ResearchResults:
        """
        Simulate tax reform with parameter uncertainty
        
        Args:
            reform: Tax reform specification
            n_monte_carlo: Number of Monte Carlo simulations
            
        Returns: ResearchResults with confidence intervals and diagnostics
        """
        # Baseline simulation
        baseline_result = self._simulate_single_reform(reform, baseline_parameters=True)
        
        # Monte Carlo for uncertainty
        mc_results = []
        for i in range(n_monte_carlo):
            # Sample parameters from uncertainty distribution
            perturbed_params = self._sample_parameters()
            
            try:
                mc_result = self._simulate_single_reform(reform, parameters=perturbed_params)
                mc_results.append(mc_result)
            except (ConvergenceError, BKConditionError) as e:
                # Log failure but continue
                logger.warning(f"Simulation {i} failed: {e}")
        
        if len(mc_results) < 0.8 * n_monte_carlo:
            raise AnalysisError(f"Too many simulations failed: {len(mc_results)}/{n_monte_carlo}")
        
        return ResearchResults(
            baseline=baseline_result,
            monte_carlo_results=mc_results,
            confidence_intervals=self._compute_confidence_intervals(mc_results),
            sensitivity_analysis=self._compute_sensitivity(mc_results),
            diagnostic_info=self._get_analysis_diagnostics()
        )
    
    def _simulate_single_reform(self, reform: TaxReform, 
                              parameters=None, baseline_parameters=False) -> SimulationResult:
        """Single reform simulation with strict validation"""
        if baseline_parameters:
            model = self.model
        else:
            # Create model with perturbed parameters
            model = self._create_perturbed_model(parameters)
        
        # Solve new steady state
        reform_steady_state = model.solve_steady_state_with_reform(reform)
        
        # Linearize and solve dynamics
        linear_system = ResearchLinearization(model, reform_steady_state).linearize()
        
        # Simulate transition path
        transition_path = self._simulate_transition_path(linear_system, reform)
        
        # Compute welfare effects
        welfare_change = self._compute_welfare_change(transition_path)
        
        return SimulationResult(
            steady_state=reform_steady_state,
            transition_path=transition_path,
            welfare_change=welfare_change,
            model_diagnostics=model.get_diagnostics()
        )
```

### B. Empirical Validation Framework

**File:** `src/empirical_validation.py` (NEW)
```python
class EmpiricalValidator:
    """Validate model results against empirical evidence"""
    
    @staticmethod
    def validate_2014_consumption_tax_reform(simulation_results: ResearchResults) -> ValidationReport:
        """
        Validate against 2014 consumption tax increase (5% → 8%)
        
        Empirical benchmarks:
        - GDP impact: -1.0% to -1.5% (Cabinet Office, 2015)
        - Consumption impact: -2.8% to -3.2% (BOJ, 2015)
        - Investment impact: -1.8% to -2.4% (OECD, 2015)
        """
        empirical_benchmarks = {
            'GDP': (-1.5, -1.0),
            'C': (-3.2, -2.8),
            'I': (-2.4, -1.8)
        }
        
        validation_results = {}
        for variable, (lower, upper) in empirical_benchmarks.items():
            simulated_impact = simulation_results.get_impact(variable, periods=4)  # 1 year
            
            if lower <= simulated_impact <= upper:
                validation_results[variable] = 'PASS'
            else:
                validation_results[variable] = f'FAIL: {simulated_impact:.2f}% not in [{lower}, {upper}]'
        
        return ValidationReport(
            test_name="2014_consumption_tax_validation",
            results=validation_results,
            overall_status='PASS' if all(v == 'PASS' for v in validation_results.values()) else 'FAIL',
            recommendations=self._get_validation_recommendations(validation_results)
        )
```

---

## 🔍 PHASE 4: RESEARCH OUTPUT FRAMEWORK (Weeks 4-5)

### A. Research-Grade Result Reporting

**File:** `src/research_reporting.py` (NEW)
```python
class ResearchReporter:
    """Generate research-grade reports with full transparency"""
    
    def generate_comprehensive_report(self, results: ResearchResults, 
                                    output_path: str) -> None:
        """Generate comprehensive research report"""
        
        report = ResearchReport()
        
        # Model specification
        report.add_section("Model Specification", self._document_model_specification())
        
        # Parameter sources and uncertainty
        report.add_section("Parameters", self._document_parameters())
        
        # Numerical methods and convergence
        report.add_section("Computational Methods", self._document_methods())
        
        # Results with confidence intervals
        report.add_section("Results", self._format_results_with_uncertainty(results))
        
        # Sensitivity analysis
        report.add_section("Sensitivity Analysis", self._generate_sensitivity_analysis())
        
        # Model limitations
        report.add_section("Limitations", self._document_limitations())
        
        # Reproducibility information
        report.add_section("Reproducibility", self._generate_reproducibility_info())
        
        report.save(output_path)
    
    def _document_limitations(self) -> str:
        """Document model limitations explicitly"""
        return """
        Model Limitations:
        
        1. **Linearization**: First-order approximation may miss important nonlinearities
        2. **Representative agent**: No heterogeneity in household responses
        3. **Closed economy**: International spillover effects not captured
        4. **Perfect foresight**: No uncertainty or learning in agent behavior
        5. **Parameter stability**: Assumes structural parameters constant over time
        
        These limitations may affect the quantitative accuracy of policy predictions.
        Results should be interpreted as conditional on model assumptions.
        """
```

---

## ⚠️ PHASE 5: MIGRATION AND SAFETY (Weeks 5-6)

### A. Gradual Migration Strategy

1. **Week 5: Parallel Implementation**
   - Create `src/research/` directory with new research-grade modules
   - Implement research classes alongside existing code
   - Create comparison tests between old and new implementations

2. **Week 6: Validation and Switchover**
   - Run comprehensive validation tests
   - Update notebooks to use research-grade classes
   - Archive old implementation with deprecation warnings

### B. Safety Measures

**File:** `src/research_mode.py` (NEW)
```python
class ResearchModeValidator:
    """Ensure research-grade requirements are met"""
    
    @staticmethod
    def enforce_research_standards(func):
        """Decorator to enforce research standards"""
        def wrapper(*args, **kwargs):
            # Check if in research mode
            if not os.environ.get('RESEARCH_MODE', False):
                raise RuntimeError(
                    "This function requires RESEARCH_MODE=True environment variable. "
                    "This ensures you are aware of the research-grade requirements."
                )
            
            # Check for proper citations
            if not hasattr(func, '_parameter_sources'):
                warnings.warn(f"{func.__name__} lacks parameter source documentation")
            
            return func(*args, **kwargs)
        return wrapper
```

---

## 📋 IMMEDIATE CHECKLIST

### This Week (Week 1):
- [ ] Implement `@RESEARCH_WARNING` decorators on all problematic functions
- [ ] Create `PARAMETER_SOURCES.md` with citation requirements
- [ ] Disable automatic fallbacks in `src/tax_simulator.py`
- [ ] Add convergence validation to all numerical methods
- [ ] Create research mode environment variable

### Critical Success Criteria:
1. **No silent failures** - All computational problems must raise exceptions
2. **No dummy data** - All results must come from actual computations or empirical sources
3. **Full transparency** - Every assumption and limitation must be documented
4. **Empirical grounding** - All parameters must have academic citations
5. **Uncertainty quantification** - All results must include confidence intervals

---

## 💡 RECOMMENDATION

Given the severity of the identified issues, I recommend:

1. **Immediate cessation** of research use until critical fixes are implemented
2. **Full audit** of any results already generated using this code
3. **Academic review** of the corrected code before publication use
4. **Collaboration** with econometricians for parameter validation

This remediation plan prioritizes research integrity over computational convenience, ensuring that the Japan Tax Simulator meets the rigorous standards required for academic and policy analysis.