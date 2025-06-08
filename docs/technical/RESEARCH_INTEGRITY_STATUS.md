# Research Integrity Status Report
## Japan Tax Simulator - Current Status and Critical Warnings

**Date:** 2025-06-08  
**Status:** 🚨 CRITICAL RESEARCH WARNINGS ACTIVE  
**Action Required:** IMMEDIATE - Review before any research use

---

## 🚨 CURRENT CRITICAL ISSUES

### ⚠️ CODE QUARANTINE STATUS
This codebase is currently **NOT SUITABLE FOR RESEARCH** without careful validation. The following critical issues have been identified:

1. **Automatic fallback mechanisms** that change underlying economic assumptions
2. **Dummy data injection** when calculations fail  
3. **Silent numerical failures** hidden by error handling
4. **Arbitrary parameter assumptions** without empirical justification
5. **Simplified welfare calculations** that ignore key economic effects

---

## 🔒 RESEARCH MODE SYSTEM

### Environment Variable Setup
```bash
# For development/testing (allows execution with warnings)
export RESEARCH_MODE=development

# For strict research use (blocks dangerous functions)
export RESEARCH_MODE=strict

# If unset, shows warning message
# unset RESEARCH_MODE
```

### Current Behavior
- **RESEARCH_MODE=development**: Functions execute but issue comprehensive warnings
- **RESEARCH_MODE=strict**: Critical functions are BLOCKED with detailed error messages
- **RESEARCH_MODE unset**: Shows configuration instructions

---

## 🚨 FUNCTIONS WITH CRITICAL WARNINGS

### 1. Tax Simulator (`src/tax_simulator.py`)

#### `EnhancedTaxSimulator.__init__()`
- **Issue**: Automatic fallback from complex to simplified model
- **Warning**: `@research_critical` - Results may change model types unexpectedly
- **Impact**: Different economic assumptions without user awareness

#### `EnhancedTaxSimulator.simulate_reform()`
- **Issue**: Automatic model selection with different assumptions
- **Warning**: `@research_critical` - Welfare calculations oversimplified
- **Impact**: Results depend on which model is selected automatically

#### `DummySteadyState` class
- **Issue**: Returns hardcoded values instead of calculations
- **Warning**: Built-in warnings on creation and dictionary conversion
- **Impact**: Results appear calculated but are actually fixed values

#### Tax breakdown estimation
- **Issue**: Arbitrary ratios (30%, 50%, 10%, 10%) for tax composition
- **Warning**: Runtime warning about lack of empirical basis
- **Impact**: Tax policy analysis based on made-up proportions

### 2. Simplified Model (`create_simple_dsge.py`)

#### `SimpleDSGEModel.simulate_tax_change()`
- **Issue**: 8-equation model lacks key macroeconomic channels
- **Warning**: `@research_critical` - Results are approximations only
- **Impact**: Missing dynamic adjustment, expectations, international effects

---

## 📊 SPECIFIC RESEARCH RISKS

### 1. **Result Contamination**
```python
# DANGEROUS: User thinks they're getting complex model results
simulator = EnhancedTaxSimulator(model)  # May silently switch to simple model
results = simulator.simulate_reform(reform)  # Uses different economic assumptions
```

### 2. **Dummy Data Injection**
```python
# DANGEROUS: Appears to be calculated but is actually hardcoded
steady_state = DummySteadyState()  # Y=1.0, C=0.6, I=0.2 (FIXED VALUES)
```

### 3. **Arbitrary Parameter Usage**
```python
# DANGEROUS: Tax composition without empirical basis
Tc = total_tax * 0.3  # 30% - WHERE DOES THIS COME FROM?
Tl = total_tax * 0.5  # 50% - NOT FROM JAPANESE DATA
```

---

## ✅ IMMEDIATE SAFETY MEASURES IMPLEMENTED

### 1. Warning Decorators Added
- `@research_critical`: Blocks function in strict mode
- `@research_deprecated`: Warns about deprecated functions
- `@research_requires_validation`: Flags empirical validation needs

### 2. Runtime Warnings Added
- DummySteadyState creation warnings
- Tax composition estimation warnings
- Model switching warnings
- Simplified model usage warnings

### 3. Documentation Updates
- `CLAUDE.md`: Added academic research requirements
- `ACADEMIC_RESEARCH_REMEDIATION_PLAN.md`: Comprehensive fix plan
- Function docstrings: Added research warnings

---

## 📋 BEFORE USING FOR RESEARCH

### Required Checklist:
- [ ] Set `RESEARCH_MODE=strict` and verify all warnings
- [ ] Review `ACADEMIC_RESEARCH_REMEDIATION_PLAN.md`
- [ ] Validate all parameters against empirical sources
- [ ] Test convergence properties explicitly
- [ ] Document all model assumptions and limitations
- [ ] Conduct sensitivity analysis for key results
- [ ] Validate results against empirical benchmarks (2014, 2019 tax reforms)

### Critical Questions to Answer:
1. **Which model is actually being used?** (Simple vs. Complex)
2. **Are steady state calculations converging properly?**
3. **What empirical data supports the parameter values?**
4. **How sensitive are results to parameter uncertainty?**
5. **Do results match known historical tax reform impacts?**

---

## 🎯 RECOMMENDED IMMEDIATE ACTIONS

### For Current Research Projects:
1. **STOP** using results until validation complete
2. **AUDIT** any results already generated
3. **VALIDATE** against empirical benchmarks
4. **DOCUMENT** all assumptions and limitations

### For Future Research:
1. **IMPLEMENT** research-grade alternatives per remediation plan
2. **VALIDATE** all parameters with empirical sources
3. **TEST** convergence and stability properties
4. **COLLABORATE** with econometricians for validation

---

## 📞 RESEARCH INTEGRITY CONTACTS

If you are using this code for:
- **Academic papers**: Consult with econometric specialists
- **Policy analysis**: Validate against government data sources
- **Thesis research**: Discuss limitations with supervisors
- **Peer review**: Full disclosure of model limitations required

---

## 📝 CURRENT WARNING EXAMPLES

When you run the code now, you'll see warnings like:

```
⚠️ RESEARCH MODE NOT SET
================================================================================
This codebase contains functions with research integrity issues.
Before using for academic research, please:
1. Review ACADEMIC_RESEARCH_REMEDIATION_PLAN.md
2. Set RESEARCH_MODE environment variable:
   - 'development' : Allow execution with warnings  
   - 'strict'      : Block risky functions entirely
3. Consider using research-grade alternatives
================================================================================

🚨 SIMPLIFIED DSGE MODEL: This model uses only 8 variables and simplified equations. 
Economic assumptions differ significantly from full DSGE models. 
Results should NOT be used for research without empirical validation.

🚨 DUMMY DATA USAGE: DummySteadyState uses hardcoded values, 
not actual economic calculations. Results are INVALID for research.
```

---

## 🔄 NEXT STEPS

1. **Week 1**: Complete critical function blocking
2. **Week 2-3**: Implement research-grade parameter validation
3. **Week 4-5**: Build empirical validation framework
4. **Week 6**: Full research-grade implementation

**Priority**: Research integrity over computational convenience.

---

**Remember**: It's better to have no result than a wrong result that could influence policy or academic conclusions.