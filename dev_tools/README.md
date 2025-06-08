# Development Tools Directory

This directory contains various development, debugging, and testing utilities for the Japan Tax Simulator project. Files have been reorganized from their original scattered locations for better project structure.

## Directory Structure

### `/blanchard_kahn_analysis/`
Scripts for analyzing and fixing Blanchard-Kahn stability conditions:
- `analyze_A_matrix.py` - Matrix structure analysis
- `check_missing_forward_terms.py` - Forward-looking variable detection
- `linearization_improved_bk_fix.py` - BK condition fixes
- `test_bk_solution.py` - BK solution testing
- `test_blanchard_kahn_fix.py` - BK fix validation
- `test_steady_state_fix.py` - Steady state solution testing

### `/debug/`
Debugging scripts for model diagnostics:
- `analyze_failure_patterns.py` - Tax reform failure analysis (moved from tests/)
- `debug_2pp_failure.py` - 2 percentage point tax failure debug (moved from tests/)
- `debug_equation_system.py` - Equation system debugging (moved from tests/)
- `debug_issue6_steady_state.py` - Issue #6 steady state debugging (moved from tests/)
- `debug_steady_state_values.py` - Steady state value debugging (moved from tests/)

### `/experimental/`
Experimental tests and development scripts:
- `simple_test.py` - Simple convergence tests (moved from tests/)
- `test_dynamics.py` - Dynamic system tests (moved from tests/)
- `test_init_guess.py` - Initial guess testing (moved from tests/)
- `test_irf_fixed.py` - IRF testing (moved from tests/)
- `test_issue6_resolution.py` - Issue resolution tests (moved from tests/)
- `test_linearization_fix.py` - Linearization fixes (moved from tests/)
- `test_simple_tax_reform.py` - Simple tax reform tests (moved from tests/)
- `test_steady_state_fix.py` - Steady state fixes (moved from tests/)
- `test_fonts.py` - Font testing for plots
- `test_integrated_simulator.py` - Integrated simulator tests
- `test_notebook_integration.py` - Notebook integration tests
- `test_notebooks.py` - Jupyter notebook testing (moved from scripts/)
- `test_tax_effects.py` - Tax effect testing

### `/calibration/`
Model calibration and parameter tuning tools:
- `diagnose_model.py` - Model diagnostic tools
- `fix_calibration.py` - Calibration fixes
- `fix_steady_state.py` - Steady state calibration

### `/legacy/`
Legacy files for reference:
- `final_irf_solution.py` - Final IRF solution (moved from tests/)

### Root Level Files
- `create_simple_dsge.py` - Simplified DSGE model implementation (moved from root)
- `README.md` - This documentation file

## Important Notes

### Research Integrity
⚠️ **CRITICAL**: Many files in this directory contain experimental code that may not be suitable for research use. Always check for research warnings and validate results against empirical data before using for policy analysis.

### Path Configuration
All Python files have been updated with proper path configurations to work from their new locations:

```python
# Standard pattern used across dev_tools files:
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now can import from src/
from src.dsge_model import DSGEModel, ModelParameters
```

### Usage
To run any script from the dev_tools directory:

```bash
# From project root
cd /Users/daisukeyoda/Documents/JapanTaxSimulator

# Run specific debugging script
uv run python dev_tools/debug/analyze_failure_patterns.py

# Run experimental tests
uv run python dev_tools/experimental/test_simple_tax_reform.py

# Run calibration tools
uv run python dev_tools/calibration/fix_steady_state.py
```

### Migration History
These files were moved as part of code organization efforts to:
1. Separate development/debugging code from production tests
2. Improve project structure and maintainability
3. Reduce clutter in the main tests/ directory
4. Group related functionality together

The main `tests/` directory now contains only proper unit and integration tests, while development utilities are organized here in `dev_tools/`.