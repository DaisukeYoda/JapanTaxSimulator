# Development Tools

This directory contains development, testing, and debugging tools for the Japan Tax Simulator DSGE model.

## Structure

### `blanchard_kahn_analysis/`
Tools and analysis files for resolving Blanchard-Kahn condition issues in DSGE linearization:

- `analyze_A_matrix.py` - Diagnostic tool for analyzing forward-looking matrix structure
- `blanchard_kahn_analysis.md` - Detailed analysis documentation of BK condition issues
- `check_missing_forward_terms.py` - Script to identify missing forward-looking terms
- `linearization_improved_bk_fix.py` - Alternative enhanced linearization implementation
- `test_bk_solution.py` - Test script for Blanchard-Kahn solution validation
- `test_blanchard_kahn_fix.py` - Comprehensive BK fix testing
- `test_steady_state_fix.py` - Steady state computation testing

## Usage

These tools are intended for:
- **Model development** - Understanding DSGE model structure and dynamics
- **Debugging** - Diagnosing linearization and stability issues
- **Research** - Analyzing different solution approaches
- **Testing** - Validating model improvements and fixes

## Note

Files in this directory are development artifacts and are excluded from the main codebase via `.gitignore`. They provide valuable reference material for future DSGE model development and troubleshooting.