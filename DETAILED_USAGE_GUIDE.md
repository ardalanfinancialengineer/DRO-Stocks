# Distributionally Robust End-to-End Portfolio Construction - Detailed Usage Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Module Documentation](#module-documentation)
4. [File Structure and Purpose](#file-structure-and-purpose)
5. [How to Use](#how-to-use)
6. [Key Concepts](#key-concepts)
7. [Running Experiments](#running-experiments)
8. [Customization Guide](#customization-guide)

---

## Project Overview

This repository implements a **Distributionally Robust (DR) End-to-End (E2E) Portfolio Construction** system as described in the paper "Distributionally Robust End-to-End Portfolio Construction" (arXiv:2206.05134).

### What Problem Does It Solve?

Traditional portfolio construction uses a two-step approach:
1. **Predict** asset returns using a model
2. **Optimize** portfolio weights based on predictions

**Problem**: The two layers are trained independently, leading to sub-optimal decisions because the prediction layer doesn't account for how errors affect portfolio decisions.

### The Solution

This project proposes an **end-to-end learning system** where:
- The prediction and decision layers are jointly trained
- The decision layer explicitly accounts for **model risk** using Distributionally Robust Optimization (DRO)
- Key parameters (risk-tolerance, degree of robustness) can be learned from data

### Key Features

- **End-to-end learning**: Information flows between prediction and decision layers
- **Model risk quantification**: Accounts for prediction errors using ambiguity sets
- **Flexible parameterization**: Learn which parameters matter for your problem
- **Multiple risk measures**: Variance or Mean Absolute Deviation
- **Multiple loss functions**: Returns, Sharpe ratio, or return-over-volatility
- **Differentiable optimization layers**: Uses cvxpylayers for gradient-based optimization

---

## System Architecture

### Two-Layer Neural Network

```
Input Features (X) 
    ↓
[PREDICTION LAYER] (Linear Neural Network)
    ↓ 
Predicted Returns (ŷ)  +  Prediction Residuals (ε)
    ↓
[DECISION LAYER] (DRO Optimization)
    ↓
Optimal Portfolio Weights (z*)
    ↓
Portfolio Performance (evaluated on future data)
```

### Decision Layer Options

The project provides three decision layer formulations:

1. **Base Model**: Pure return maximization
   - Minimize: `-ŷ @ z`
   - Constraints: sum(z) = 1, z ≥ 0

2. **Nominal Model**: Risk-return trade-off
   - Minimize: `(1/n_obs) * Σ risk(z, ε_t) - γ * ŷ @ z`
   - Parameters: γ (risk-return trade-off)
   - Constraints: sum(z) = 1, z ≥ 0

3. **Distributionally Robust (DR) Models**: Robust to distributional uncertainty
   - Minimize: `worst-case risk - γ * ŷ @ z` over ambiguity set
   - Parameters: γ (risk-return trade-off), δ (robustness degree)
   - Distance metrics: Hellinger or Total Variation (TV)
   - Constraints: sum(z) = 1, z ≥ 0

---

## Module Documentation

### `e2edro.py` - Core E2E Learning System

**Purpose**: Defines the main end-to-end neural network architecture with differentiable optimization layers.

**Key Classes and Functions**:

#### CvxpyLayer Functions (Optimization Formulations)

| Function | Purpose | Parameters |
|----------|---------|-----------|
| `base_mod()` | Pure return maximization layer | `n_y`, `n_obs`, `prisk` |
| `nominal()` | Risk-return trade-off layer | `n_y`, `n_obs`, `prisk` |
| `tv()` | DRO with Total Variation distance | `n_y`, `n_obs`, `prisk` |
| `hellinger()` | DRO with Hellinger distance | `n_y`, `n_obs`, `prisk` |

#### Main Class: `e2e_net`

```python
e2e_net(n_x, n_y, n_obs, opt_layer='nominal', prisk='p_var', 
        perf_loss='sharpe_loss', pred_loss_factor=0.5, perf_period=13,
        train_pred=True, train_gamma=True, train_delta=True, set_seed=None)
```

**Key Methods**:

- `forward(X, Y)`: Forward pass through prediction and optimization layers
  - Returns: optimal weights (z_star) and predicted returns (y_hat)

- `net_cv(X, Y, lr_list, epoch_list)`: Cross-validation on validation set
  - Tests different learning rates and epoch counts
  - Stores results in `cv_results` DataFrame

- `net_roll_test(X, Y, n_roll=4)`: Rolling window out-of-sample test
  - Simulates real portfolio rebalancing
  - Stores results in `portfolio` object

**Key Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `train_pred` | bool | Learn prediction layer weights (θ) |
| `train_gamma` | bool | Learn risk-tolerance parameter (γ) |
| `train_delta` | bool | Learn robustness parameter (δ) |
| `opt_layer` | str | 'base_mod', 'nominal', 'hellinger', or 'tv' |
| `prisk` | str | Risk function: 'p_var' (variance) or 'p_mad' (MAD) |
| `perf_loss` | str | Performance metric: 'sharpe_loss', 'single_period_loss', etc. |
| `pred_loss_factor` | float | Weight for prediction MSE in total loss (0-1) |

---

### `RiskFunctions.py` - Risk Measures

**Purpose**: Defines portfolio risk measures used in optimization.

**Available Risk Functions**:

| Function | Formula | Description |
|----------|---------|-------------|
| `p_var(z, c, x)` | (x @ z - c)² | Squared deviation from mean (portfolio variance) |
| `p_mad(z, c, x)` | \|x @ z - c\| | Absolute deviation from mean (MAD) |

**Usage Note**: These functions return single-scenario risk. They must be aggregated over all scenarios to compute total portfolio risk.

---

### `LossFunctions.py` - Performance Loss Functions

**Purpose**: Defines loss functions for evaluating out-of-sample portfolio performance.

**Available Loss Functions**:

| Function | Formula | Description |
|----------|---------|-------------|
| `single_period_loss()` | -y_perf[0] @ z | Negative return (next period) |
| `single_period_over_var_loss()` | -y_perf[0] @ z / std(y_perf @ z) | Negative return-to-volatility ratio |
| `sharpe_loss()` | -mean(y_perf @ z) / std(y_perf @ z) | Negative Sharpe ratio (simplified) |

**Note**: These are used to train the E2E system. Negative values make them minimization objectives.

---

### `PortfolioClasses.py` - Data and Results Storage

**Purpose**: Defines objects for managing datasets and storing backtest results.

#### `SlidingWindow` Class

A PyTorch Dataset for creating rolling windows from time series data.

```python
SlidingWindow(X, Y, n_obs, perf_period)
```

**Provides**:
- Sliding windows of features (X) with window size n_obs + 1
- Corresponding asset returns (Y) with window size n_obs
- Forward-looking performance data (y_perf) for evaluation

#### `backtest` Class

Stores out-of-sample portfolio performance metrics.

```python
backtest(len_test, n_y, dates)
```

**Attributes**:
- `weights`: Asset weights per period (len_test × n_y)
- `rets`: Realized portfolio returns per period
- `dates`: Corresponding dates
- `stats()`: Method to compute summary statistics
  - `mean`: Average return (annualized)
  - `vol`: Return volatility (standard deviation)
  - `sharpe`: Sharpe ratio (mean/vol)
  - `tri`: Total Return Index (cumulative wealth)

#### `InSample` Class

Tracks training metrics during E2E learning.

**Attributes**:
- `loss`: Training loss per epoch
- `gamma`: Risk-tolerance parameter values during training
- `delta`: Robustness parameter values during training

---

### `BaseModels.py` - Baseline Comparison Models

**Purpose**: Implements naive models for comparison.

#### `equal_weight` Class

Simple equal-weight portfolio (1/n allocation to each asset).

#### `pred_then_opt` Class

Traditional "predict-then-optimize" baseline.

```python
pred_then_opt(n_x, n_y, n_obs, set_seed=None, prisk='p_var', opt_layer='nominal')
```

**Key Methods**:
- `forward(X, Y)`: Returns optimal weights and predictions
- `net_roll_test(X, Y, n_roll=4)`: Out-of-sample backtest
- `net_cv()`: Cross-validation (if applicable)

**Key Difference from E2E**: Prediction and optimization layers are trained separately.

---

### `DataLoad.py` - Data Loading and Synthetic Data Generation

**Purpose**: Provides utilities for loading real market data and generating synthetic datasets.

#### `TrainTest` Class

Manages train/validation/test splits.

```python
TrainTest(data, n_obs, split)
```

**Methods**:
- `train()`: Returns training subset
- `test()`: Returns test subset (with overlap for continuity)
- `split_update(split)`: Dynamically update split ratios

#### Data Generation Functions

| Function | Purpose |
|----------|---------|
| `AV()` | Download real asset data from AlphaVantage API |
| `synthetic()` | Generate linear synthetic data: Y = a + X @ b + X2 @ c + noise |
| `synthetic_nl()` | Generate nonlinear synthetic data with quadratic terms |
| `synthetic_NN()` | Generate synthetic data using random 3-layer neural network |
| `KF()` | Load Kenneth French factor data |

#### Statistical Analysis

- `statanalysis(X, Y)`: Computes correlation and statistical significance between features and returns

---

### `PlotFunctions.py` - Visualization

**Purpose**: Creates publication-quality plots and tables.

**Key Functions**:

| Function | Output |
|----------|--------|
| `wealth_plot()` | Portfolio wealth evolution (total return index) over time |
| `sr_bar()` | Bar chart comparing Sharpe ratios across portfolios |
| `fin_table()` | Summary statistics table (return, volatility, Sharpe ratio) |
| `turnover_plot()` | Portfolio rebalancing activity |

---

## File Structure and Purpose

```
E2E-DRO-main/
├── main.py                          # Main experimental script (Experiments 1-4)
├── README.md                        # Original project README
├── DETAILED_USAGE_GUIDE.md         # This file
├── LICENSE                          # Apache 2.0 License
│
├── e2edro/                          # Core module
│   ├── __init__.py                 # Package initialization
│   ├── e2edro.py                   # Main E2E neural network class
│   ├── RiskFunctions.py            # Portfolio risk measures (variance, MAD)
│   ├── LossFunctions.py            # Performance loss functions (Sharpe, etc.)
│   ├── PortfolioClasses.py         # Data containers (SlidingWindow, backtest, InSample)
│   ├── BaseModels.py               # Baseline models (equal weight, predict-then-optimize)
│   ├── DataLoad.py                 # Data loading and synthetic data generation
│   ├── PlotFunctions.py            # Visualization utilities
│   └── __pycache__/                # Python cache directory
│
├── cache/                           # Cached results and models
│   ├── exp/                         # Experiment 1-4 results
│   │   ├── dro_initial_state_*     # Saved model files
│   │   ├── nom_initial_state_*     # Saved model files
│   │   ├── dr_net/                 # DRO model variants
│   │   ├── dr_net_learn_delta/     # DRO models with learnable δ
│   │   ├── dr_net_learn_gamma/     # DRO models with learnable γ
│   │   ├── dr_net_learn_theta/     # DRO models with learnable θ
│   │   └── plots/                  # Generated plots (PDF)
│   └── exp5/                        # Experiment 5 results (2-3 layer networks)
```

---

## How to Use

### Installation

**Dependencies**:
```
Python 3.x
numpy
scipy
pandas
matplotlib
cvxpy 1.x
cvxpylayers 0.1.x
PyTorch 1.x
pandas_datareader
alpha_vantage
```

**Install via pip**:
```bash
pip install torch numpy scipy pandas matplotlib cvxpy cvxpylayers pandas_datareader alpha_vantage
```

### Quick Start Example

#### 1. Generate or Load Data

```python
from e2edro import DataLoad as dl

# Generate synthetic data
X, Y = dl.synthetic(n_x=10, n_y=20, n_tot=1500, n_obs=104, split=[0.6, 0.4])

# Or load real data from AlphaVantage
# X, Y = dl.AV(start='2000-01-01', end='2021-09-30', split=[0.6, 0.4], 
#              freq='weekly', n_obs=104, n_y=20, use_cache=True)
```

#### 2. Create an E2E Model

```python
from e2edro import e2edro as e2e

# Create end-to-end DRO model with learnable parameters
model = e2e.e2e_net(
    n_x=10,                          # Number of features
    n_y=20,                          # Number of assets
    n_obs=104,                       # Window size for residuals
    opt_layer='hellinger',           # DRO layer with Hellinger distance
    prisk='p_var',                   # Use variance risk measure
    perf_loss='sharpe_loss',         # Optimize Sharpe ratio
    pred_loss_factor=0.5,            # Balance prediction + performance loss
    perf_period=13,                  # Look-ahead period (weeks)
    train_pred=True,                 # Learn prediction weights
    train_gamma=True,                # Learn risk tolerance
    train_delta=True,                # Learn robustness degree
    set_seed=1000                    # For reproducibility
).double()  # Use double precision
```

#### 3. Run Cross-Validation

```python
# Find best hyperparameters on validation set
lr_list = [0.005, 0.0125, 0.02]      # Learning rates to try
epoch_list = [30, 40, 50, 60, 80, 100] # Epoch counts to try

model.net_cv(X, Y, lr_list, epoch_list)

# View results
print(model.cv_results)
```

#### 4. Out-of-Sample Testing

```python
# Rolling window backtest on test set
model.net_roll_test(X, Y, n_roll=4)

# Access results
portfolio = model.portfolio
print(f"Mean return: {portfolio.mean:.2%}")
print(f"Volatility: {portfolio.vol:.2%}")
print(f"Sharpe ratio: {portfolio.sharpe:.2f}")
```

#### 5. Visualize Results

```python
from e2edro import PlotFunctions as pf

# Compare multiple models
portfolios = [baseline_model.portfolio, proposed_model.portfolio]
names = ['Baseline', 'E2E-DRO']
colors = ['gray', 'green']

pf.wealth_plot(portfolios, names, colors, path='wealth_evolution.pdf')
pf.sr_bar(portfolios, names, colors, path='sharpe_comparison.pdf')
```

---

## Key Concepts

### Parameter Learning

The E2E system learns three types of parameters:

#### 1. **θ (Theta)** - Prediction Weights
- Weights in the linear prediction layer
- Maps features to predicted returns: ŷ = X @ θ
- Learned when `train_pred=True`

#### 2. **γ (Gamma)** - Risk-Return Trade-off
- Controls preference between expected return and risk
- Higher γ → seek returns more aggressively (more risk)
- Lower γ → seek stability (less risk)
- Learned when `train_gamma=True`
- Often initialized uniformly in [0.02, 0.1]

#### 3. **δ (Delta)** - Robustness/Ambiguity Size
- Controls how much uncertainty to account for
- Higher δ → more robust (assumes wider range of possible distributions)
- Lower δ → more aggressive (assumes tighter distribution range)
- Learned when `train_delta=True`
- Range depends on distance metric and n_obs

### Loss Function Components

The total training loss combines two terms:

```
Total Loss = pred_loss_factor * MSE_loss + (1 - pred_loss_factor) * perf_loss
```

**MSE Loss**: 
- How well the prediction layer fits the data
- `MSE = mean((Y - Ŷ)²)`

**Performance Loss**:
- How well the portfolio performs on out-of-sample data
- `perf_loss = -sharpe_loss` or `-single_period_loss`, etc.

**Trade-off**:
- `pred_loss_factor=1.0`: Pure prediction optimization (traditional)
- `pred_loss_factor=0.5`: Balanced between prediction and performance
- `pred_loss_factor=0.0`: Pure performance optimization (end-to-end)

### Ambiguity Sets

The DRO models optimize over distributions within an **ambiguity set**:

**Hellinger Distance**:
- Controls probability deviations from empirical distribution
- δ bounds the Hellinger divergence
- More suitable when δ should be normalized
- Implemented in `hellinger()` function

**Total Variation Distance**:
- Measures maximum probability mass difference
- δ bounds the TV divergence
- More stringent robustness guarantee
- Implemented in `tv()` function

### Rolling Window Testing

The `net_roll_test()` method simulates real portfolio management:

```
Training Period 1  → Optimization → Test Period 1
Training Period 2  → Optimization → Test Period 2
Training Period 3  → Optimization → Test Period 3
Training Period 4  → Optimization → Test Period 4
```

This ensures:
- Models are retrained regularly (simulating practice)
- No look-ahead bias
- Performance is evaluated out-of-sample
- Parameter evolution is tracked across periods

---

## Running Experiments

### Main Experiments Script

The `main.py` file implements 4-5 main experiments from the paper:

#### **Experiment 1: General Comparison**

Compares 5 approaches on the same data:

```python
# Models tested:
ew_net           # Equal-weight baseline
po_net           # Predict-then-optimize (nominal)
base_net         # E2E with base model
nom_net          # E2E with nominal model (learn θ, γ)
dr_net           # E2E with DRO model (learn θ, γ, δ)
```

**Configuration**:
```python
perf_loss = 'sharpe_loss'              # Optimize Sharpe ratio
perf_period = 13                       # 13 weeks ahead
pred_loss_factor = 0.5                 # Balanced loss
prisk = 'p_var'                        # Variance risk
dr_layer = 'hellinger'                 # Hellinger distance
```

#### **Experiment 2: Learning the Robustness Degree (δ)**

Studies the value of learning δ versus fixing it:

```python
dr_po_net            # Predict-then-optimize with fixed δ
dr_net_learn_delta   # E2E with learnable δ (fixed θ)
```

Hypothesis: E2E with learned δ outperforms because it optimizes δ for portfolio performance, not just prediction accuracy.

#### **Experiment 3: Learning Risk Appetite (γ)**

Studies whether learning γ improves performance:

```python
nom_net_learn_gamma     # Nominal E2E (learn γ, fixed θ)
dr_net_learn_gamma      # DRO E2E (learn γ, fixed θ)
dr_net_learn_gamma_delta # DRO E2E (learn γ, δ, fixed θ)
```

Hypothesis: Learned γ better matches market risk preferences than fixed values.

#### **Experiment 4: Learning Prediction Weights (θ)**

Studies the value of E2E joint training:

```python
nom_net_learn_theta  # Nominal E2E (learn θ, fixed γ)
dr_net_learn_theta   # DRO E2E (learn θ, fixed γ)
```

Hypothesis: End-to-end training of θ for portfolio performance beats standard prediction training.

### Running main.py

**With Cached Models** (uses pre-saved models):
```bash
python main.py  # Set use_cache=True in the script
```

**From Scratch** (trains all models):
```bash
# Edit main.py and set: use_cache=False
python main.py
```

**Custom Experiments** (modify hyperparameters):
```python
# Edit these parameters in main.py:
perf_loss = 'sharpe_loss'              # Change to 'single_period_loss'
perf_period = 13                       # Change lookahead period
pred_loss_factor = 0.5                 # Adjust loss trade-off
lr_list = [0.005, 0.0125, 0.02]       # Test different learning rates
epoch_list = [30, 40, 50, 60, 80, 100] # Test different epoch counts
```

### Output Files

Results are saved in `./cache/exp/`:

**Model Files**:
- `*.pkl`: Trained model objects (can be loaded with pickle)

**Plots**:
- `plots/wealth_exp*.pdf`: Wealth evolution charts
- `plots/sr_bar_exp*.pdf`: Sharpe ratio comparisons

**Data**:
- `cv_results`: Cross-validation results DataFrame
- `portfolio`: Backtest results object

---

## Customization Guide

### Creating a Custom Risk Function

Add to `RiskFunctions.py`:

```python
def p_custom_risk(z, c, x):
    """Custom risk measure
    Inputs
    z: Portfolio weights (n_y x 1)
    c: Centering parameter (scalar)
    x: Scenario returns (n_y x 1)
    
    Output: Risk measure for this scenario
    """
    # Example: Conditional Value-at-Risk proxy
    return cp.maximum(x @ z - c, 0)**2
```

Then use it in `e2e_net`:
```python
model = e2e.e2e_net(..., prisk='p_custom_risk')
```

### Creating a Custom Loss Function

Add to `LossFunctions.py`:

```python
def custom_loss(z_star, y_perf):
    """Custom performance loss
    Inputs
    z_star: Optimal weights
    y_perf: Forward-looking realizations
    
    Output: Scalar loss to minimize
    """
    return -torch.mean(y_perf @ z_star)  # Negative for minimization
```

Then use it in `e2e_net`:
```python
model = e2e.e2e_net(..., perf_loss='custom_loss')
```

### Using Different Prediction Models

The current implementation uses linear predictions. To use neural networks:

**Modify `e2edro.py`**:

```python
# In e2e_net.__init__(), replace:
self.pred_layer = nn.Linear(n_x, n_y)

# With:
self.pred_layer = nn.Sequential(
    nn.Linear(n_x, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, n_y)
)
```

### Changing Data Frequency

In `main.py`, modify:
```python
freq = 'weekly'    # Change to 'daily', 'monthly', etc.
```

### Using Different Distance Metrics

```python
dr_layer = 'tv'    # Use Total Variation instead of Hellinger
# Or
dr_layer = 'hellinger'  # Use Hellinger distance
```

### Asset Selection

```python
# In main.py:
n_y = 20  # Number of assets to include

# Modify data loading if using real data:
X, Y = dl.AV(start, end, split, freq=freq, n_obs=n_obs, 
             n_y=n_y, use_cache=True, AV_key=AV_key)
```

---

## Troubleshooting

### Common Issues

**1. CUDA/GPU Errors**
```python
# main.py automatically handles this:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**2. Convergence Issues**
- Try smaller learning rates: `lr_list = [0.001, 0.005, 0.01]`
- Try more epochs: `epoch_list = [50, 100, 150]`
- Adjust `pred_loss_factor` toward 1.0 for stability

**3. Memory Issues**
- Reduce `n_obs` (window size)
- Reduce `n_y` (number of assets)
- Use single precision: remove `.double()`

**4. API Key Issues (AlphaVantage)**
```python
# Register at alphavantage.co
AV_key = "your_api_key_here"
X, Y = dl.AV(..., AV_key=AV_key)
```

### Debugging

**Print Model Parameters**:
```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data}")
```

**Check Training Progress**:
```python
# After net_cv()
print(model.cv_results)  # View validation losses
print(f"Best gamma: {model.gamma_trained}")
print(f"Best delta: {model.delta_trained}")
```

**Inspect Portfolio Performance**:
```python
portfolio = model.portfolio
print(f"Dates: {portfolio.rets.index}")
print(f"Returns: {portfolio.rets['rets']}")
print(f"Weights: {portfolio.weights}")
```

---

## References

- **Paper**: [Distributionally Robust End-to-End Portfolio Construction](https://arxiv.org/abs/2206.05134)
- **Authors**: Giorgio Costa, Garud Iyengar
- **Institution**: Columbia University IEOR Department
- **cvxpylayers**: [GitHub](https://github.com/cvxgrp/cvxpylayers)
- **cvxpy**: [Documentation](https://www.cvxpy.org/)

---

## Citation

If you use this code, please cite:

```bibtex
@article{costa2022distributionally,
  title={Distributionally robust end-to-end portfolio construction},
  author={Costa, Giorgio and Iyengar, Garud},
  journal={arXiv preprint arXiv:2206.05134},
  year={2022}
}
```

---

## License

Apache 2.0 License - See LICENSE file for details.
