# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Japan Tax Simulator project that implements a Dynamic Stochastic General Equilibrium (DSGE) model for analyzing tax policy impacts on the Japanese economy. The model simulates consumption, income, capital, and corporate tax changes with full transition dynamics.

## Common Development Commands

### Environment Setup

```bash
# Install uv if not already installed
brew install uv  # macOS
# or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
uv sync

# Verify installation
uv run python quick_check.py
```

### Testing and Validation
```bash
# Quick system check
uv run python quick_check.py

# Run all tests using pytest
uv run pytest tests/

# Run specific test categories
uv run pytest tests/unit/          # Unit tests
uv run pytest tests/integration/   # Integration tests

# Run debug scripts
uv run python scripts/debug/debug_linearization.py
uv run python scripts/validation/compare_linearizations.py

# Start Jupyter for interactive analysis
uv run jupyter notebook
```

### Running Simulations
```bash
# Basic simulation via Python script
uv run python -c "
from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import TaxPolicySimulator, TaxReform

params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)
simulator = TaxPolicySimulator(model)
"

# Interactive notebooks
uv run jupyter notebook notebooks/tax_simulation_demo.ipynb
```

## Architecture

### Core Model Structure
The model follows a standard DSGE framework with four main economic sectors:
- **Household Sector**: Consumption, labor supply, savings decisions with habit formation
- **Firm Sector**: Production with Calvo price stickiness and investment adjustment costs  
- **Government Sector**: Fiscal policy with debt stabilization rules
- **Central Bank**: Taylor rule monetary policy

### Key Components

**src/dsge_model.py**
- `ModelParameters`: Central parameter management class
- `SteadyState`: Steady state values container
- `DSGEModel`: Main model class with steady state computation and equation system

**src/tax_simulator.py** 
- `TaxReform`: Tax policy specification (permanent/temporary/phased)
- `EnhancedTaxSimulator`: Full transition dynamics simulation
- `SimulationResults`: Results container with welfare analysis

**src/linearization_improved.py**
- `ImprovedLinearizedDSGE`: Advanced linearization using Klein (2000) method
- Handles Blanchard-Kahn conditions for unique stable solutions

### Data Flow
1. Parameters loaded from `config/parameters.json`
2. Model initialized and steady state computed
3. Model linearized around steady state
4. Tax reforms simulated with transition dynamics
5. Results exported with welfare analysis

### Tax Policy Modeling
The simulator supports four tax instruments:
- **$\tau_c$**: Consumption tax (VAT)
- **$\tau_l$**: Labor income tax
- **$\tau_k$**: Capital income tax  
- **$\tau_f$**: Corporate tax

Implementation modes:
- **Permanent**: Tax rate change maintained indefinitely
- **Temporary**: Tax change for specified duration then reverts
- **Phased**: Gradual implementation over multiple periods

### International Extension
The model includes open economy features:
- Real exchange rate (q)
- Net foreign assets (b_star) 
- Import/export functions with price elasticities
- Uncovered interest parity condition

### Calibration
Parameters are calibrated to Japanese economic data:
- Quarterly frequency ($\beta=0.99 \approx$ 4% annual discount rate)
- Tax rates match recent Japanese levels (consumption tax 10%, etc.)
- Macro ratios target Japanese national accounts data

## Working with the Code

### Parameter Modifications
Edit `config/parameters.json` to adjust model calibration. Key sections:
- `model_parameters`: Behavioral parameters ($\sigma_c$, $\sigma_l$, $\alpha$, etc.)
- `tax_parameters.baseline`: Current tax rates
- `calibration_targets`: Target ratios for steady state

### Adding New Tax Scenarios
```python
reform = TaxReform(
    name="Custom Reform",
    tau_c=0.12,  # 12% consumption tax
    tau_l=0.18,  # 18% income tax
    implementation='phased',
    phase_in_periods=8  # 2 years gradual implementation
)
```

### Steady State Convergence Issues
If steady state computation fails:
1. Check parameter bounds (especially ensure $\beta < 1$, tax rates < 1)
2. Verify Taylor principle ($\phi_\pi > 1$ for monetary policy)
3. Try different solver methods in `compute_steady_state()`
4. Check fiscal sustainability (debt-to-GDP bounds)

### Model Validation
Always verify:
- Blanchard-Kahn condition satisfied (unique stable solution)
- Steady state ratios match calibration targets
- Impulse responses have reasonable signs and magnitudes
- No explosive paths in simulations

## File Structure Context

- **config/**: Model parameters and calibration targets
- **src/**: Core model implementation modules
- **tests/**: Test files organized by category
  - **tests/unit/**: Unit tests for individual components
  - **tests/integration/**: Full system integration tests
- **scripts/**: Utility scripts for development and analysis
  - **scripts/debug/**: Debugging and diagnostic scripts
  - **scripts/validation/**: Model validation and comparison scripts
  - **scripts/examples/**: Example usage scripts
- **docs/**: Project documentation (reorganized June 2025)
  - **docs/USER_GUIDE.md**: Comprehensive user guide (formerly COMPREHENSIVE_DOCUMENTATION.md)
  - **docs/EXAMPLES.md**: Working code examples (formerly ACCURATE_EXAMPLES.md)
  - **docs/REFACTORING_SUMMARY.md**: Refactoring completion summary
  - **docs/technical/**: Technical documentation and theory
  - **docs/development/**: Development guides and setup instructions
  - **docs/research/**: Policy research and independent fiscal institutions studies
- **notebooks/**: Interactive Jupyter demonstrations and analysis
- **results/**: Output files from simulation runs
- **data/**: Supporting data files (if any)
- **README.md**: Concise project overview (Japanese)
- **README_EN.md**: English version of project overview

The model is designed for policy analysis, so prioritize economic interpretability and robustness over computational speed. Always validate results against economic intuition and existing literature.

## Linearization Method Selection (Issue #30 Resolution)

**⚠️ CRITICAL FOR RESEARCH**: As of June 2025, Issue #30 analysis revealed significant accuracy differences between linearization methods:

### Method Comparison Results
- **83% of scenarios show >5% difference** between simplified and full linearization
- **Maximum difference: 7.54%** (income tax reduction scenario)
- **Only small reforms (<2pp)** show acceptable differences

### Usage Guidelines

**🎓 Academic Research & Policy Analysis:**
```python
simulator = EnhancedTaxSimulator(
    model, 
    use_simple_model=False,
    use_simple_linearization=False  # REQUIRED for research
)
```

**📚 Demonstrations & Education:**
```python
simulator = EnhancedTaxSimulator(
    model,
    use_simple_model=False, 
    use_simple_linearization=True   # Stable, easy to understand
)
```

**🔍 Robustness Testing:**
```python
# Run both methods and compare results
# Report differences if >5% for transparency
```

### Technical Details
- **Simplified**: Fixed coefficients, always stable, fast computation
- **Full Klein**: DSGE-derived, Blanchard-Kahn conditions, theoretical rigor
- **Decision threshold**: 5% relative difference for significance

### Documentation
- Full analysis: `docs/technical/LINEARIZATION_METHOD_GUIDE.md`
- Comparison tool: `scripts/validation/linearization_method_comparison.py`
- Test suite: `scripts/validation/test_linearization_options.py`

**Academic Integrity**: Always specify the linearization method in publications. Default behavior (auto-selection) triggers research warnings.

# CRITICAL ACADEMIC RESEARCH REQUIREMENTS

**⚠️ WARNING: This is a RESEARCH CODEBASE for academic and policy analysis.**

## Academic Integrity Requirements (MANDATORY)

### 1. NO DUMMY VALUES OR FALLBACKS
- **NEVER** use placeholder values (0.0, 1.0, etc.) when real data is unavailable
- **NEVER** return fallback results when models fail to converge
- **NEVER** create "DummySteadyState" or similar placeholder objects
- **NEVER** estimate tax breakdowns without empirical data sources

### 2. NO FABRICATED PERFORMANCE METRICS OR BENCHMARKS
- **NEVER** make up execution times, memory usage, or performance statistics
- **NEVER** invent system requirements without actual testing
- **NEVER** fabricate benchmark comparisons or speed claims
- **ALWAYS** measure and report actual performance metrics
- **ALWAYS** specify the test environment (hardware, OS, Python version)
- **ALWAYS** include performance variability and failure modes
- **Example of PROHIBITED practice**: "Simulation takes 10-30 seconds" without measurement
- **Example of REQUIRED practice**: "Measured 0.01 seconds on M1 Pro macOS, may fail with Blanchard-Kahn warnings"

### 3. FAIL FAST AND EXPLICIT
```python
# ✅ CORRECT: Explicit failure
if convergence_failed:
    raise ConvergenceError("Model failed to converge. Check parameter bounds and fiscal sustainability.")

# ❌ WRONG: Silent fallback
if convergence_failed:
    return default_steady_state  # DANGEROUS for research
```

### 4. DATA SOURCE REQUIREMENTS
- All parameters must cite specific empirical sources
- Tax data: Ministry of Finance Annual Reports
- Macro data: Cabinet Office National Accounts  
- Behavioral parameters: Published academic estimates with citations

### 5. EXPLICIT ASSUMPTION DOCUMENTATION
```python
def compute_tax_elasticity(self):
    """
    Computes tax elasticity using:
    - Labor supply elasticity: 2.0 (Keane & Rogerson, 2012)  
    - Consumption elasticity: 1.5 (Ogaki & Reinhart, 1998)
    
    WARNING: Results sensitive to these parameter values.
    Conduct sensitivity analysis before policy conclusions.
    """
```

### 5. UNCERTAINTY AND ROBUSTNESS
- Provide confidence intervals for all estimates
- Conduct sensitivity analysis for key parameters
- Report model limitations explicitly
- Validate against empirical benchmarks

## Prohibited Practices in Research Code

### ❌ NEVER DO THIS:
```python
# Silent failure handling
try:
    result = complex_computation()
except:
    return 0.0  # DANGEROUS

# Dummy data creation  
baseline_data['Tc'] = [total_tax * 0.3] * periods  # Arbitrary assumption

# Hidden assumptions
welfare_change = 0.0  # Default when calculation fails
```

### ✅ REQUIRED APPROACH:
```python
# Explicit failure with diagnostic information
def compute_steady_state(self, max_iterations=1000, tolerance=1e-8):
    """
    Returns: SteadyState object with empirically grounded values
    Raises: ConvergenceError with detailed diagnostic information
    """
    for i in range(max_iterations):
        if converged:
            return self._validate_steady_state(result)
    
    raise ConvergenceError(
        f"Failed to converge after {max_iterations} iterations. "
        f"Final residual: {residual:.2e}. "
        f"Check: 1) Parameter bounds, 2) Fiscal sustainability, 3) BK conditions"
    )

# Empirically grounded data only
def get_tax_composition(self, data_year: int, source: str):
    """
    Args:
        data_year: Year of fiscal data (e.g., 2019)
        source: "MOF_Annual_Report" or specific empirical source
    
    Raises: 
        NotImplementedError if empirical data not available
        ValueError if data_year not in valid range
    """
    if source not in self.validated_sources:
        raise NotImplementedError(f"Tax composition requires empirical data from {source}")
```

## Research Quality Checklist

Before any analysis, verify:

- [ ] All parameters have empirical citations
- [ ] Model convergence is verified (no silent failures)
- [ ] Steady state ratios match Japanese data
- [ ] Blanchard-Kahn conditions satisfied
- [ ] Sensitivity analysis conducted
- [ ] Results validated against literature
- [ ] Uncertainty bounds provided
- [ ] Limitations clearly stated

## Error Handling Philosophy

**Research Principle**: Better to have no result than a wrong result that could influence policy or academic conclusions.

- **Computational errors**: Stop execution with diagnostic information
- **Data availability**: Explicit error, suggest empirical data sources  
- **Model failures**: Detailed convergence diagnostics, parameter suggestions
- **Assumption violations**: Clear warnings with literature citations

Remember: Academic and policy credibility depends on rigorous methodology, not just working code.

## Documentation Structure

### Main Documentation Files
- **README.md** - Project overview and quick start guide
- **docs/USER_GUIDE.md** - Comprehensive user guide and API reference (formerly COMPREHENSIVE_DOCUMENTATION.md)
- **docs/EXAMPLES.md** - Working code examples and scenarios (formerly ACCURATE_EXAMPLES.md)
- **docs/REFACTORING_SUMMARY.md** - Modular architecture implementation summary

### Documentation Organization
- **docs/development/** - Developer setup and troubleshooting guides
- **docs/technical/** - Technical specifications, linearization methods, architecture
- **docs/research/** - Policy research, fiscal institution analysis, methodology
- **docs/planning/** - Future development plans and integration roadmaps

### File Reference Guidelines
When helping users navigate the codebase:
- Direct new users to **README.md** for quick start
- Refer to **docs/USER_GUIDE.md** for comprehensive API usage
- Use **docs/EXAMPLES.md** for working code examples
- Point to **docs/technical/** for implementation details
- Reference **docs/research/** for policy analysis background
