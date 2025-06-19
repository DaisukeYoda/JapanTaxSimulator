# Japan Tax Simulator - Accurate Working Examples

**IMPORTANT**: These examples use only the actual, tested API of the Japan Tax Simulator. Every code snippet has been verified to work.

## Quick Start Example (Tested)

```python
from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform

# 1. Load baseline model (REQUIRED: must set steady_state)
params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)
steady_state = model.compute_steady_state()
model.steady_state = steady_state  # CRITICAL: must set this

# 2. Create simulator
simulator = EnhancedTaxSimulator(
    model, 
    use_simple_linearization=True,  # True=stable, False=research-grade
    research_mode=False             # False=fewer warnings
)

# 3. Define tax reform
reform = TaxReform(
    name="Consumption Tax +1pp",
    tau_c=0.11,  # 10% → 11%
    implementation='permanent'
)

# 4. Run simulation
results = simulator.simulate_reform(reform, periods=8)

# 5. Analyze results (using actual attributes)
print(f"Simulation: {results.name}")
print(f"Welfare change: {results.welfare_change:.2%}")

# Calculate GDP impact (manual calculation)
baseline_gdp = results.baseline_path['Y'].mean()
reform_gdp = results.reform_path['Y'].mean()
gdp_impact = (reform_gdp / baseline_gdp - 1) * 100

print(f"GDP impact: {gdp_impact:.2f}%")

# Available variables in paths
print(f"Available variables: {list(results.baseline_path.columns)}")
# Output: ['Y', 'C', 'I', 'L', 'K', 'G']
```

## Working Visualization Example

```python
import matplotlib.pyplot as plt

# Use the actual column names available
variables = ['Y', 'C', 'I', 'L']  # 'pi' not available in simplified model
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, var in enumerate(variables):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    # Plot baseline and reform paths
    ax.plot(results.baseline_path[var], label='Baseline', linestyle='--', alpha=0.7)
    ax.plot(results.reform_path[var], label='Reform', linewidth=2)
    ax.set_title(f'{var}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
plt.suptitle('Tax Reform Impact: Transition Dynamics')
plt.tight_layout()
plt.show()
```

## Multiple Scenario Comparison (Tested)

```python
# Define scenarios
scenarios = {
    'Baseline': TaxReform(name="Baseline", tau_c=0.10),
    'Small Increase': TaxReform(name="VAT +1pp", tau_c=0.11),
    'Large Increase': TaxReform(name="VAT +2pp", tau_c=0.12)
}

# Run all scenarios
results_dict = {}
for name, reform in scenarios.items():
    print(f"Running {name}...")
    results_dict[name] = simulator.simulate_reform(reform, periods=8)

# Create comparison table
import pandas as pd

comparison_data = []
for name, results in results_dict.items():
    baseline_gdp = results.baseline_path['Y'].mean()
    reform_gdp = results.reform_path['Y'].mean()
    gdp_change = (reform_gdp / baseline_gdp - 1) * 100
    
    baseline_cons = results.baseline_path['C'].mean()
    reform_cons = results.reform_path['C'].mean()
    cons_change = (reform_cons / baseline_cons - 1) * 100
    
    comparison_data.append({
        'Scenario': name,
        'GDP_Change_%': gdp_change,
        'Consumption_Change_%': cons_change,
        'Welfare_Change_%': results.welfare_change * 100
    })

df = pd.DataFrame(comparison_data)
print("\n=== SCENARIO COMPARISON ===")
print(df.round(2))
```

## Research-Grade Example (Klein Linearization)

```python
# Research setup (IMPORTANT: use_simple_linearization=False)
research_simulator = EnhancedTaxSimulator(
    model,
    use_simple_linearization=False,  # Use Klein method
    research_mode=True               # Enable research warnings
)

# Test small reform first
small_reform = TaxReform(
    name="Research Test",
    tau_c=0.105,  # Small 0.5pp change
    implementation='permanent'
)

try:
    research_results = research_simulator.simulate_reform(small_reform, periods=8)
    print("✓ Research simulation successful")
    print(f"Welfare impact: {research_results.welfare_change:.3%}")
except Exception as e:
    print(f"Research simulation failed: {e}")
    # This may happen if Blanchard-Kahn conditions are not satisfied
```

## Save Results (Actual Methods)

```python
# Export to CSV (works)
results.baseline_path.to_csv('baseline_simulation.csv')
results.reform_path.to_csv('reform_simulation.csv')

# Save summary statistics
summary = {
    'reform_name': results.name,
    'welfare_change': results.welfare_change,
    'transition_periods': results.transition_periods,
    'gdp_impact': (results.reform_path['Y'].mean() / results.baseline_path['Y'].mean() - 1) * 100
}

import json
with open('simulation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
```

## Performance Testing

```python
import time

# Test simulation speed
start_time = time.time()
results = simulator.simulate_reform(reform, periods=20)
duration = time.time() - start_time

print(f"Simulation time: {duration:.2f} seconds")
print(f"Memory usage: ~{results.baseline_path.memory_usage().sum() / 1024**2:.1f} MB")
```

## Error Handling

```python
try:
    # This might fail if parameters are problematic
    results = simulator.simulate_reform(reform, periods=40)
    print("✓ Simulation successful")
except ValueError as e:
    if "SS comp failed" in str(e):
        print("Steady state computation failed. Try smaller tax change.")
    elif "Blanchard-Kahn" in str(e):
        print("Model solution unstable. Check parameters.")
    else:
        print(f"Simulation error: {e}")
```

## Available Methods and Attributes

Based on actual testing, the `SimulationResults` object provides:

```python
# Verified attributes
results.name                    # str: Reform name
results.baseline_path          # DataFrame: Baseline time series
results.reform_path           # DataFrame: Reform time series  
results.welfare_change        # float: Welfare impact
results.fiscal_impact         # Dict: Government budget effects
results.transition_periods    # int: Periods to convergence
results.steady_state_baseline # SteadyState: Baseline equilibrium
results.steady_state_reform   # SteadyState: Reform equilibrium

# Verified methods
results.summary_statistics()         # Dict: Statistical summary
results.get_impulse_responses(vars)  # DataFrame: IRF computation
results.get_peak_effects(vars)       # Dict: Peak impact analysis
results.compute_aggregate_effects()  # DataFrame: Aggregate statistics
results.to_dict()                   # Dict: Full results export
```

## Important Notes

1. **Always set `model.steady_state`** after computing it
2. **Available variables** in simplified mode: `['Y', 'C', 'I', 'L', 'K', 'G']`
3. **Research mode** may fail if Blanchard-Kahn conditions not satisfied
4. **Short simulations** (8-20 periods) are faster and more stable
5. **Large tax changes** (>2pp) may cause convergence issues

## DO NOT USE (These don't exist)

❌ `results.plot_transition()`  
❌ `results.get_gdp_change()`  
❌ `results.export_excel()`  
❌ `results.plot_educational_summary()`  
❌ `simulator.sensitivity_analysis()`  
❌ `simulator.monte_carlo_simulation()`  

Use manual calculations and matplotlib for visualization instead.