# Federated Learning Playground

A modular Python framework for experimenting with Federated Learning (FL) algorithms, with built-in support for encryption schemes, custom aggregation strategies, and comprehensive visualization tools.

## Features

- **Modular Architecture** - Clean separation between Party, Orchestrator, Model, Encryption, and Aggregation components
- **Extensible Design** - Easy to add custom models, encryption schemes, or aggregation strategies via abstract base classes
- **Simulation Environment** - High-level `Environment` class for quick experimentation
- **Rich Visualizations** - Built-in tools for training history, model comparisons, and divergence analysis
- **Flexible Data Loading** - Support for CSV files, pandas DataFrames, or numpy arrays. Handles both standard and transposed formats (features as rows)
- **Master Thesis Integration** - Compatible with master thesis data format and experiments. Load genomics datasets (Metabric) seamlessly
- **Encryption Ready** - Interface for homomorphic encryption schemes (with `NoEncryption` baseline)
- **Analytics** - Track local and global model performance across training rounds

## Installation

### Prerequisites

- Python >= 3.13
- `uv` package manager (recommended) or `pip`

### Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd /path/to/fed_env
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   
   Using `uv` (recommended):
   ```bash
   uv pip install -e .
   ```
   
   Using `pip`:
   ```bash
   pip install -e .
   ```

## Quick Start

```python
from fed_playground import (
    Environment,
    ClosedFormLinearRegressionModel,
    NoEncryption,
    MeanAggregation,
    DataLoader
)

# Load your data
loader = DataLoader(file_path='data.csv', target_column='target')

# Create federated learning environment
env = Environment(
    n_parties=5,                                    # Number of participants
    encryption_scheme=NoEncryption(),               # No encryption (baseline)
    aggregation_strategy=MeanAggregation(),         # Simple averaging
    model_class=ClosedFormLinearRegressionModel,   # Closed-form linear regression
    data_loader=loader
)

# Run simulation
history = env.run_simulation(rounds=10)

# Access results
print(f"Final global loss: {history['global_loss'][-1]}")
```

## Architecture

The framework consists of several modular components that work together:

```
┌──────────────────────────────────────────────────────────┐
│                     Environment                          │
│  (Orchestrates the entire FL simulation)                 │
└───────────┬────────────────────────────┬─────────────────┘
            │                            |
    ┌───────▼────────┐            ┌──────▼──────┐
    │  Orchestrator  │            │   Parties   │
    │  (Aggregator)  │◄───────────┤   (Local    │
    └───────┬────────┘            │   Trainers) │
            │                     └──────┬──────┘
            │                            │
    ┌───────▼────────┐             ┌─────▼───────┐
    │  Aggregation   │             │    Models   │
    │   Strategy     │             │  (ML Logic) │
    └────────────────┘             └─────────────┘
```

## Core Components

### 📦 [`fed_playground/src/`](file:///Users/cola/Desktop/Projects/fed_env/fed_playground/src)

#### [`environment.py`](file:///Users/cola/Desktop/Projects/fed_env/fed_playground/src/environment.py)
High-level orchestrator that sets up parties, manages data distribution, and runs FL rounds.

**Key Methods:**
- `setup()` - Initializes parties, data splits, and orchestrator
- `run_round()` - Executes one round of FL (broadcast → train → aggregate)
- `run_simulation(rounds, test_data)` - Runs complete simulation, returns history

#### [`party.py`](file:///Users/cola/Desktop/Projects/fed_env/fed_playground/src/party.py)
Represents a single participant in federated learning.

**Key Methods:**
- `train_local_model()` - Trains on private local data
- `get_encrypted_model()` - Returns encrypted parameters
- `update_model(global_params)` - Updates from global model

#### [`orchestrator.py`](file:///Users/cola/Desktop/Projects/fed_env/fed_playground/src/orchestrator.py)
Central coordinator that aggregates model updates.

**Key Methods:**
- `distribute_model()` - Sends global model to all parties
- `aggregate_models()` - Collects and aggregates local updates

#### [`models.py`](file:///Users/cola/Desktop/Projects/fed_env/fed_playground/src/models.py)
Machine learning model implementations.

**Available Models:**
- `LinearRegressionModel` - Gradient descent based linear regression
- `ClosedFormLinearRegressionModel` - Closed-form solution using normal equations (faster, exact)

**Interface:**
- `train(X, y)` - Train on data
- `get_parameters()` / `set_parameters()` - Parameter access
- `evaluate(X, y)` - Compute MSE loss
- `predict(X)` - Make predictions

#### [`aggregation.py`](file:///Users/cola/Desktop/Projects/fed_env/fed_playground/src/aggregation.py)
Strategies for combining model updates.

**Available:**
- `MeanAggregation` - Simple averaging (FedAvg)

**Interface:**
- `aggregate(encrypted_models, encryption_scheme)` - Combine updates

#### [`encryption.py`](file:///Users/cola/Desktop/Projects/fed_env/fed_playground/src/encryption.py)
Encryption scheme abstractions.

**Available:**
- `NoEncryption` - Passthrough for baseline comparisons

**Interface:**
- `encrypt(params)` - Encrypt model parameters
- `decrypt(encrypted)` - Decrypt parameters
- `aggregate(encrypted_list)` - Homomorphic aggregation

#### [`dataloader.py`](file:///Users/cola/Desktop/Projects/fed_env/fed_playground/src/dataloader.py)
Flexible data loading from multiple sources.

**Supported Inputs:**
1. **CSV File**: `DataLoader(file_path='data.csv', target_column='target')`
2. **Pandas DataFrame**: `DataLoader(dataframe=df, target_column='target')`
3. **NumPy Arrays**: `DataLoader(X=X_array, y=y_array)`

#### [`visualization.py`](file:///Users/cola/Desktop/Projects/fed_env/fed_playground/src/visualization.py)
Comprehensive visualization toolkit (see [Visualization](#-visualization) section).

#### [`utils_data.py`](file:///Users/cola/Desktop/Projects/fed_env/fed_playground/src/utils_data.py)
Utilities for synthetic data generation and splitting.

## Usage Examples

### Basic Simulation with Synthetic Data

```python
from fed_playground import Environment, NoEncryption, MeanAggregation, LinearRegressionModel

# Environment will generate synthetic data internally
env = Environment(
    n_parties=3,
    n_features=5,
    n_samples=200,
    encryption_scheme=NoEncryption(),
    aggregation_strategy=MeanAggregation(),
    model_class=LinearRegressionModel,
    model_params={'learning_rate': 0.01, 'epochs': 10}
)

history = env.run_simulation(rounds=5)
```

### Loading Custom CSV Data

```python
from fed_playground import Environment, DataLoader, ClosedFormLinearRegressionModel, NoEncryption, MeanAggregation

# Prepare data
loader = DataLoader(
    file_path='my_data.csv',
    target_column='target',
    feature_columns=['feature_0', 'feature_1', 'feature_2']  # Optional
)

env = Environment(
    n_parties=4,
    encryption_scheme=NoEncryption(),
    aggregation_strategy=MeanAggregation(),
    model_class=ClosedFormLinearRegressionModel,
    data_loader=loader
)

history = env.run_simulation(rounds=10)
```

### Using NumPy Arrays Directly

```python
import numpy as np
from fed_playground import DataLoader, Environment, NoEncryption, MeanAggregation, ClosedFormLinearRegressionModel

# Your data
X = np.random.randn(1000, 5)
y = np.random.randn(1000)

loader = DataLoader(X=X, y=y)

env = Environment(
    n_parties=5,
    encryption_scheme=NoEncryption(),
    aggregation_strategy=MeanAggregation(),
    model_class=ClosedFormLinearRegressionModel,
    data_loader=loader
)

history = env.run_simulation(rounds=10)
```

### Custom Test Data

```python
# Use custom test set for evaluation
X_test, y_test = load_test_data()

history = env.run_simulation(rounds=10, test_data=(X_test, y_test))
```

## 📊 Visualization

The framework includes a modular visualization system for analyzing FL experiments.

### Available Visualizers

#### 1. **TrainingHistoryVisualizer** - Training Metrics Over Rounds

```python
from fed_playground import TrainingHistoryVisualizer

# Run simulation
history = env.run_simulation(rounds=10)

# Visualize training history
viz = TrainingHistoryVisualizer(save_dir="results")
viz.plot(
    data={"Global Loss": history["global_loss"]},
    title="Training Progress",
    xlabel="Round",
    ylabel="MSE Loss",
    filename="training_history.png"
)
```

**Output:** Line plot showing how metrics evolve across rounds.

#### 2. **ComparisonVisualizer** - Model Performance Comparison

```python
from fed_playground import ComparisonVisualizer

comparison_data = {
    "Centralized": 0.15,
    "Federated": 0.18,
    "Local (Party 0)": 0.25,
}

viz = ComparisonVisualizer(save_dir="results")
viz.plot(
    data=comparison_data,
    title="Model Performance Comparison",
    xlabel="Model Type",
    ylabel="Test MSE",
    filename="comparison.png",
    color=['green', 'blue', 'orange']
)
```

**Output:** Bar chart comparing different models or configurations.

#### 3. **DivergenceVisualizer** - Federated vs Centralized Analysis

Analyze how federated learning diverges from centralized training across different experimental parameters.

```python
from fed_playground import DivergenceVisualizer
import numpy as np

visualizer = DivergenceVisualizer(save_dir="results")

# Run experiments varying number of parties
for n_parties in [2, 4, 6, 8]:
    # Train centralized model
    centralized_model = ClosedFormLinearRegressionModel(input_dim=5)
    centralized_model.train(X_train, y_train)
    gen_mse = centralized_model.evaluate(X_test, y_test)
    
    # Run federated learning
    metrics_per_round = []
    for round_num in range(5):
        env = Environment(n_parties=n_parties, ...)
        env.setup()
        env.run_round()
        
        # Evaluate federated model
        fed_model = ClosedFormLinearRegressionModel(input_dim=5)
        fed_model.set_parameters(env.orchestrator.global_model_params)
        fed_mse = fed_model.evaluate(X_test, y_test)
        
        # Calculate divergence metrics
        w_fed = fed_model.get_parameters()
        w_gen = centralized_model.get_parameters()
        
        round_metrics = {
            "mse": fed_mse,
            "general_mse": gen_mse,
            "normdiff": np.linalg.norm(w_fed - w_gen),
            "msediff": fed_mse - gen_mse,
            "mseratio": fed_mse / gen_mse,
            # Optional: local model metrics
            "local_mse": [party.model.evaluate(X_test, y_test) for party in env.parties],
            "local_normdiff": [np.linalg.norm(party.model.get_parameters() - w_gen) for party in env.parties],
        }
        metrics_per_round.append(round_metrics)
    
    visualizer.add_result(n_parties, metrics_per_round)

# Generate plots
visualizer.plot(
    x_label="Number of Parties",
    title_suffix="Party Count",
    show_local_models=True
)
```

**Output:** Three plots analyzing divergence:
- `norm_difference.png` - Parameter norm differences
- `mse_difference.png` - MSE differences (federated - centralized)
- `mse_ratio.png` - MSE ratios (federated / centralized)

Each plot includes:
- Boxplots for federated and local model distributions
- Line plots for MSE values
- Dual y-axes for different scales

## Advanced Usage

### Divergence Analysis Script

The [`examples/divergence_analysis.py`](file:///Users/cola/Desktop/Projects/fed_env/examples/divergence_analysis.py) script provides three analysis modes:

#### 1. **Instances Diff** - Vary number of parties

```bash
python plot_divergence.py \
    --data-path test_data.csv \
    --features feature_0 feature_1 feature_2 \
    --target target \
    --instances-diff \
    --min-instances 2 \
    --max-instances 10 \
    --step-instances 2 \
    --data-per-instance 50 \
    --rounds 5 \
    --save-path results
```

#### 2. **Data Diff** - Vary data per party

```bash
python plot_divergence.py \
    --data-path test_data.csv \
    --features feature_0 feature_1 \
    --target target \
    --data-diff \
    --min-data 10 \
    --max-data 100 \
    --step-data 10 \
    --instances 5 \
    --rounds 5
```

#### 3. **Fixed Data** - Fixed total data, distributed among varying parties

```bash
python plot_divergence.py \
    --data-path test_data.csv \
    --features feature_0 feature_1 \
    --target target \
    --fixed-data \
    --total-data-points 500 \
    --min-instances 2 \
    --max-instances 10 \
    --rounds 5
```

### Creating Custom Components

#### Custom Model

```python
from fed_playground import Model
import numpy as np

class MyCustomModel(Model):
    def __init__(self, input_dim: int):
        self.params = np.random.randn(input_dim + 1)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # Your training logic
        pass
    
    def get_parameters(self) -> np.ndarray:
        return self.params
    
    def set_parameters(self, params: np.ndarray) -> None:
        self.params = params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Your prediction logic
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

# Use in Environment
env = Environment(model_class=MyCustomModel, ...)
```

#### Custom Aggregation Strategy

```python
from fed_playground import AggregationStrategy
import numpy as np

class WeightedAggregation(AggregationStrategy):
    def __init__(self, weights):
        self.weights = weights  # Weight per party
    
    def aggregate(self, encrypted_models, encryption_scheme):
        # Decrypt models
        models = [encryption_scheme.decrypt(m) for m in encrypted_models]
        
        # Weighted average
        weighted_sum = sum(w * m for w, m in zip(self.weights, models))
        return weighted_sum / sum(self.weights)

# Use in Environment
env = Environment(aggregation_strategy=WeightedAggregation([1, 2, 1]), ...)
```

#### Custom Encryption Scheme

```python
from fed_playground import EncryptionScheme
import numpy as np

class SimpleAdditiveMasking(EncryptionScheme):
    def __init__(self):
        self.mask = None
    
    def encrypt(self, params: np.ndarray):
        self.mask = np.random.randn(*params.shape)
        return params + self.mask
    
    def decrypt(self, encrypted_params):
        return encrypted_params - self.mask
    
    def aggregate(self, encrypted_params_list):
        # Masks cancel out when summing
        return sum(encrypted_params_list)

# Use in Environment
env = Environment(encryption_scheme=SimpleAdditiveMasking(), ...)
```

## Master Thesis Data Integration

The library supports data in transposed format (features as rows, samples as columns), which is used in the master thesis and genomics datasets like Metabric.

### Loading Transposed Data

```python
from fed_playground import DataLoader

# Load data with genes/features as rows (master thesis format)
loader = DataLoader(
    file_path='/path/to/Metabric.csv',
    target_column='BRCA1',  # Gene to predict
    feature_columns=['TP53', 'MKI67', 'FOXM1', 'KIF20A'],  # Predictor genes
    transpose=True  # Indicate transposed format
)

X, y = loader.load()  # Returns data in standard format (samples × features)
```

### Thesis Integration Demo

Run experiments matching master thesis methodology:

```bash
# Using synthetic data
python examples/thesis_integration_demo.py \
    --data synthetic \
    --target y \
    --features x1 x2 x3 x4 \
    --instances-diff \
    --rounds 5

# Using Metabric data
python examples/thesis_integration_demo.py \
    --data metabric \
    --target BRCA1 \
    --features TP53 MKI67 FOXM1 KIF20A \
    --data-diff

# List all master thesis results
python examples/thesis_integration_demo.py --list-results
```

### Thesis Utilities

Helper functions for master thesis data compatibility:

```python
from examples.thesis_utils import (
    load_thesis_csv,
    parse_thesis_experiment_name,
    get_thesis_data_path
)

# Load thesis data
data = load_thesis_csv('/path/to/data.csv')

# Get path to preset datasets
synthetic_path = get_thesis_data_path('synthetic')
metabric_path = get_thesis_data_path('metabric')

# Parse thesis experiment directory names
params = parse_thesis_experiment_name(
    'instances_diff_y_x1-x2-x3-x4_max_15_min_2_samples_8_rounds_20'
)
print(params['mode'])      # 'instances_diff'
print(params['target'])    # 'y'
print(params['features'])  # ['x1', 'x2', 'x3', 'x4']
```

## Examples

See complete working examples in [`examples/`](file:///Users/cola/Desktop/Projects/fed_env/examples):

- [`visualization_demo.py`](file:///Users/cola/Desktop/Projects/fed_env/examples/visualization_demo.py) - Comprehensive demonstration of all visualization tools
- [`divergence_analysis.py`](file:///Users/cola/Desktop/Projects/fed_env/examples/divergence_analysis.py) - Full divergence analysis with multiple modes
- [`thesis_integration_demo.py`](file:///Users/cola/Desktop/Projects/fed_env/examples/thesis_integration_demo.py) - Master thesis data integration and compatibility
- [`thesis_utils.py`](file:///Users/cola/Desktop/Projects/fed_env/examples/thesis_utils.py) - Helper functions for thesis data

Run the visualization demo:
```bash
python examples/visualization_demo.py
```

This generates sample plots in the `demo_results/` directory.

## Project Structure

```
fed_env/
├── fed_playground/           # Main package
│   ├── __init__.py          # Public API exports
│   └── src/                 # Source modules
│       ├── aggregation.py   # Aggregation strategies
│       ├── dataloader.py    # Data loading utilities
│       ├── encryption.py    # Encryption schemes
│       ├── environment.py   # FL simulation orchestrator
│       ├── models.py        # ML model implementations
│       ├── orchestrator.py  # Central aggregator
│       ├── party.py         # FL participant
│       ├── utils_data.py    # Data generation/splitting
│       └── visualization.py # Visualization toolkit
├── examples/                # Usage examples and test data
│   ├── README.md            # Examples documentation
│   ├── basic_simulation.py  # Basic FL simulation
│   ├── divergence_analysis.py  # Divergence analysis
│   ├── visualization_demo.py   # Visualization demos
│   └── test_data.csv        # Sample dataset
├── pyproject.toml           # Project configuration
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Dependencies

- `numpy` >= 2.4.0 - Numerical computing
- `pandas` >= 2.3.3 - Data manipulation
- `matplotlib` >= 3.10.8 - Visualization
- `tqdm` >= 4.67.1 - Progress bars

## Troubleshooting

### ModuleNotFoundError

**Issue:** `ModuleNotFoundError: No module named 'fed_playground'`

**Solution:** Install the package in editable mode:
```bash
pip install -e .
```

### Data Loading Errors

**Issue:** `ValueError: Must provide either file_path, dataframe, or (X, y)`

**Solution:** Ensure DataLoader receives exactly one input source:
```python
# Good
loader = DataLoader(file_path='data.csv')
loader = DataLoader(X=X_array, y=y_array)

# Bad - providing multiple sources
loader = DataLoader(file_path='data.csv', X=X_array, y=y_array)
```

### CSV File Not Found

**Issue:** `FileNotFoundError` when running examples

**Solution:** Test data is included in `examples/test_data.csv`. If needed, you can generate new data:

```python
import numpy as np
import pandas as pd

np.random.seed(42)
X = np.random.randn(200, 5)
true_weights = np.random.randn(5)
y = np.dot(X, true_weights) + 0.5 + np.random.normal(0, 0.1, 200)

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
df['target'] = y
df.to_csv('examples/test_data.csv', index=False)
```

### Memory Issues with Large Datasets

**Issue:** Out of memory errors with many parties or large datasets

**Solution:**
- Reduce `n_samples` or `n_parties`
- Use `ClosedFormLinearRegressionModel` instead of `LinearRegressionModel` (more efficient)
- Process data in batches if implementing custom models

## Contributing

We welcome contributions! To extend the framework:

1. **Models**: Inherit from `Model` in `models.py`
2. **Aggregation**: Inherit from `AggregationStrategy` in `aggregation.py`
3. **Encryption**: Inherit from `EncryptionScheme` in `encryption.py`
4. **Visualizers**: Inherit from `Visualizer` in `visualization.py`

Follow the existing code patterns and include docstrings.

## FAQ

**Q: Can I use this for real-world federated learning?**

A: This is a **playground** for experimentation and research. For production FL, consider frameworks like TensorFlow Federated or PySyft.

**Q: How do I add my own model architecture?**

A: Inherit from the `Model` abstract base class and implement `train()`, `get_parameters()`, `set_parameters()`, `predict()`, and `evaluate()`. See [Custom Model](#custom-model) example.

**Q: Does this support GPU acceleration?**

A: Not currently. This is a NumPy-based framework focused on simplicity and educational use.

**Q: How do I implement secure aggregation?**

A: Implement a custom `EncryptionScheme` (e.g., using a homomorphic encryption library) and `AggregationStrategy` that works with encrypted parameters.

**Q: Can I save/load trained models?**

A: Use `model.get_parameters()` to extract parameters and save with `np.save()`. Load with `np.load()` and restore with `model.set_parameters()`.

---

**Happy Experimenting!**
