# Blanchard-Kahn Condition Analysis for DSGE Model

## Problem Summary
The DSGE model linearization is failing the Blanchard-Kahn conditions because:
- **A matrix rank is only 5** (should be much higher)
- Only **2 explosive eigenvalues** but **5 forward-looking variables**
- Only **6 out of 29 equations** contain forward-looking terms

## Key Issues Identified

### 1. Low A Matrix Rank (5 vs 27)
The A matrix captures forward-looking dynamics (coefficients on E[x_{t+1}]). With rank 5, it means:
- Only 5 independent forward-looking relationships
- System is effectively degenerate in forward-looking dimension
- Cannot determine unique rational expectations solution

### 2. Limited Forward-Looking Variables
Only 5 variables appear with forward expectations:
- `C` (consumption)
- `Lambda` (marginal utility)
- `Rk_gross` (capital return)
- `pi_gross` (inflation)
- `q` (real exchange rate)

### 3. Missing Forward-Looking Behavior
Important variables that typically have forward-looking behavior in DSGE models but don't here:
- `Y` (output) - No forward expectations despite New Keynesian features
- `I` (investment) - Should depend on future capital returns
- `K` (capital) - Capital accumulation equation is backward-looking only
- `w` (wages) - No wage stickiness or forward-looking wage setting
- `mc` (marginal cost) - Affects current inflation but no forward expectations

## Root Causes

### 1. Model Specification Issues
- The model may be missing key forward-looking relationships
- No forward-looking IS curve (output gap doesn't depend on expected future output)
- Investment decision is static rather than forward-looking
- No forward-looking wage or price setting beyond basic Phillips curve

### 2. Linearization Issues
The linearization correctly identifies forward-looking terms, but:
- Two equations (1 and 15) were dropped due to low norm
- These might have contained important forward-looking relationships
- The system is overdetermined (29 equations, 27 variables)

### 3. Economic Interpretation
The model appears to be:
- More backward-looking than typical DSGE models
- Missing intertemporal optimization conditions
- Lacking forward-looking behavior in production/investment decisions

## Recommendations

### 1. Check Model Equations
Review the model specification to ensure:
- Investment Euler equation includes expected future returns
- Output gap equation includes expected future output
- Capital accumulation includes forward-looking elements
- Labor market has forward-looking wage setting

### 2. Verify Equation Dropping
The linearization drops 2 equations - verify these aren't critical:
- Check which equations are being dropped
- Ensure no forward-looking relationships are lost

### 3. Consider Model Extensions
To increase forward-looking behavior:
- Add habit formation in investment
- Include adjustment costs that depend on future investment
- Add forward-looking terms in government spending rule
- Include expected future foreign variables in trade equations

### 4. Alternative Solution Methods
If Blanchard-Kahn conditions cannot be satisfied:
- Use generalized Schur decomposition with different sorting
- Try undetermined coefficients method
- Consider whether model has indeterminacy (multiple equilibria)
- Check if model needs additional forward-looking constraints

## Technical Details

### A Matrix Structure
```
Non-zero elements: 8 (out of 729 possible)
Forward-looking variables: C, Lambda, Rk_gross, pi_gross, q
Equations with forward terms: 2, 4, 5, 7, 13, 20
```

### Eigenvalue Analysis
- Need: Number of explosive eigenvalues = Number of forward-looking variables
- Have: 2 explosive eigenvalues, 5 forward-looking variables
- Result: Indeterminacy (multiple equilibria possible)

This analysis suggests the model needs structural modifications to achieve a unique rational expectations equilibrium.