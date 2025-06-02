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
- **τc**: Consumption tax (VAT)
- **τl**: Labor income tax
- **τk**: Capital income tax  
- **τf**: Corporate tax

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
- Quarterly frequency (β=0.99 ≈ 4% annual discount rate)
- Tax rates match recent Japanese levels (consumption tax 10%, etc.)
- Macro ratios target Japanese national accounts data

## Working with the Code

### Parameter Modifications
Edit `config/parameters.json` to adjust model calibration. Key sections:
- `model_parameters`: Behavioral parameters (σc, σl, α, etc.)
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
1. Check parameter bounds (especially ensure β < 1, tax rates < 1)
2. Verify Taylor principle (φπ > 1 for monetary policy)
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
- **docs/**: Project documentation
  - **docs/technical/**: Technical documentation and theory
  - **docs/development/**: Development guides and setup instructions
- **notebooks/**: Interactive Jupyter demonstrations and analysis
- **results/**: Output files from simulation runs
- **data/**: Supporting data files (if any)

The model is designed for policy analysis, so prioritize economic interpretability and robustness over computational speed. Always validate results against economic intuition and existing literature.