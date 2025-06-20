# Japan Tax Simulator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Dynamic Stochastic General Equilibrium (DSGE) model for analyzing the macroeconomic impacts of tax policy changes on the Japanese economy.

English | [日本語](README.md)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/DaisukeYoda/JapanTaxSimulator.git
cd JapanTaxSimulator

# Install dependencies using uv
uv sync

# Verify installation
uv run python quick_check.py
```

### Basic Usage Example

```python
from src.dsge_model import DSGEModel, ModelParameters
from src.tax_simulator import EnhancedTaxSimulator, TaxReform

# Initialize model
params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)
steady_state = model.compute_steady_state()
model.steady_state = steady_state  # Important!

# Tax reform simulation
simulator = EnhancedTaxSimulator(model)
reform = TaxReform(
    name="VAT +1pp",
    tau_c=0.11,  # 10% → 11%
    implementation='permanent'
)

results = simulator.simulate_reform(reform, periods=40)
print(f"Welfare change: {results.welfare_change:.2%}")
```

## 📖 Key Features

- **4 Tax Instruments**: Consumption, labor income, capital income, and corporate taxes
- **Dynamic Simulation**: Analyze short-term and long-term economic effects
- **Welfare Analysis**: Evaluate social welfare changes from policy reforms
- **Research-Grade Precision**: Rigorous model suitable for academic research

## 📚 Documentation

### User Guides
- **[USER_GUIDE.md](docs/USER_GUIDE.md)** - Comprehensive usage guide, API reference, configuration
- **[EXAMPLES.md](docs/EXAMPLES.md)** - Working code examples and scenario analysis
- **[development/](docs/development/)** - Developer information (setup, troubleshooting)
- **[technical/](docs/technical/)** - Technical specifications, theoretical background, architecture
- **[research/](docs/research/)** - Policy research, independent fiscal institutions studies

## 🔧 Model Features

### Economic Sectors
- **Households**: Consumption-labor optimization (with habit formation)
- **Firms**: Calvo price setting, investment adjustment costs
- **Government**: Fiscal rules, debt stabilization mechanisms
- **Central Bank**: Taylor rule, inflation targeting

### Tax Analysis
| Tax Type | Baseline | Analysis Focus |
|----------|----------|----------------|
| Consumption Tax | 10% | VAT macroeconomic impacts |
| Income Tax | 20% | Labor supply incentive effects |
| Capital Income Tax | 25% | Saving and investment behavior |
| Corporate Tax | 30% | Business investment decisions |

## 📊 Analysis Example

### Impact of 1pp Consumption Tax Increase

| Variable | Short-term (1 year) | Long-term (steady state) |
|----------|---------------------|--------------------------|
| GDP | -0.8% | -0.3% |
| Consumption | -1.2% | -0.6% |
| Investment | -0.4% | -0.1% |
| Welfare change | -0.15% | -0.15% |

*Note: Values are example simulation results. Actual results vary with parameter settings.*

## 🔧 Installation and Setup

### System Requirements
- Python 3.11+
- uv (recommended) or pip

### Using uv (Recommended)

```bash
# Install uv if not already installed
# macOS
brew install uv
# Other OS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Project setup
git clone https://github.com/DaisukeYoda/JapanTaxSimulator.git
cd JapanTaxSimulator
uv sync

# Run commands (examples)
uv run python quick_check.py
uv run jupyter notebook
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📋 Usage Examples and Scenarios

```python
# Phased implementation example
phased_reform = TaxReform(
    name="Gradual VAT Increase",
    tau_c=0.15,  # 10% → 15%
    implementation='phased',
    phase_in_periods=12  # 3-year gradual implementation
)

# Temporary policy example
temporary_reform = TaxReform(
    name="Temporary Income Tax Cut",
    tau_l=0.15,  # 20% → 15%
    implementation='temporary',
    duration=8  # 2 years only
)

# Results visualization
results.baseline_path.plot(y='Y', label='Baseline')
results.reform_path.plot(y='Y', label='Reform')
plt.title('GDP Impact')
plt.legend()
plt.show()
```

### Jupyter Notebook Execution

```bash
# Start notebook
uv run jupyter notebook

# Try these demos:
# notebooks/tax_simulation_demo.ipynb - Basic usage
# notebooks/policy_analysis_demo.ipynb - Policy analysis examples
```

## 👥 Community and Support

### Questions and Feedback
- **GitHub Issues**: [https://github.com/DaisukeYoda/JapanTaxSimulator/issues](https://github.com/DaisukeYoda/JapanTaxSimulator/issues)
- **Discussions**: GitHub Discussions for questions and idea exchange

### Contributing
1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Create a pull request

### License
This project is published under the [MIT License](LICENSE).

## 📚 References

### Theoretical Background
- Galí, J. (2015). "Monetary Policy, Inflation, and the Business Cycle"
- Woodford, M. (2003). "Interest and Prices"

### Japanese Economy Applications
- Cabinet Office ESRI (2018) "Macro Econometric Model"
- Bank of Japan (2020) "Quarterly Japanese Economic Model (Q-JEM)"

---

*Note: This model is developed for simulation and analysis purposes. Actual policy decisions require additional verification and expert evaluation.*