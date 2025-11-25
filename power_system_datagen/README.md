# Power System Data Generation Framework

A standalone, physics-based data generation framework for power system transient analysis using synchronous generator models. This framework generates high-quality training data for machine learning applications in power systems.

## Features

- **Physics-Based Modeling**: Accurate synchronous generator dynamics with classical machine model
- **Flexible Sampling**: Multiple sampling strategies for time, states, and voltage profiles
- **ODE Simulation**: High-precision trajectory simulation using SciPy/PyTorch
- **Configurable**: Easy-to-use configuration system for different dataset types
- **Visualization**: Built-in plotting tools for data analysis
- **IEEE Test Systems**: Pre-configured parameters for IEEE 9-bus system generators

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy >= 1.20.0`
- `scipy >= 1.7.0`
- `torch >= 1.10.0`
- `matplotlib >= 3.3.0`

Optional (for PyTorch ODE backend):
- `torchdiffeq >= 0.2.0`

### Setup

```bash
# Clone or copy the framework
cd power_system_datagen

# Install dependencies
pip install numpy scipy torch matplotlib

# Optional: Install torchdiffeq for PyTorch backend
pip install torchdiffeq
```

## Quick Start

### Basic Usage

```python
from power_system_datagen import DataGenerator
from power_system_datagen.config import DatasetConfig

# Create configuration
config = DatasetConfig(
    n_samples=1000,
    t_min=0.0,
    t_max=0.5,
    time_sampling="linear",
    state_sampling="annular",
    simulate=True,
    random_seed=42
)

# Initialize generator with IEEE 9-bus Gen 1 parameters
generator = DataGenerator("ieee9_1", config)

# Generate dataset
dataset = generator.generate_dataset(verbose=True)

# Save dataset
generator.save(dataset, "my_dataset.pkl")
```

### Generate Standard Datasets

```python
from power_system_datagen.generator import generate_standard_datasets

# Generate train/test/collocation datasets
datasets = generate_standard_datasets(
    generator_id="ieee9_1",
    output_dir="./data",
    verbose=True
)
```

## Framework Architecture

```
power_system_datagen/
├── models/              # Power system component models
│   ├── generator_model.py      # Synchronous generator dynamics
│   └── voltage_profile.py      # Polynomial voltage profiles
├── sampling/            # Sampling strategies
│   └── samplers.py             # Time, state, and voltage sampling
├── simulation/          # ODE simulation
│   └── simulator.py            # ODE solver wrapper
├── config/              # Configuration
│   ├── generator_params.py     # Generator parameters
│   └── dataset_config.py       # Dataset configuration
├── utils/               # Utilities
│   ├── data_io.py              # Save/load functions
│   └── visualization.py        # Plotting tools
├── examples/            # Usage examples
└── generator.py         # Main data generation orchestrator
```

## Core Components

### 1. Synchronous Generator Model

Classical synchronous machine model with 4 states:
- **E_q'**: Transient EMF in q-axis
- **E_d'**: Transient EMF in d-axis
- **delta**: Rotor angle (rad)
- **omega**: Angular frequency deviation (p.u.)

**Control inputs**:
- **P_M**: Mechanical power (p.u.)
- **E_fd**: Field voltage (p.u.)

**Key equations**:
```
dE_q'/dt = 0  (fast dynamics neglected)
dE_d'/dt = 0
d(delta)/dt = omega * omega_s
d(omega)/dt = (P_M - P_e - D*omega) / (2*H)
```

### 2. Voltage Profile

Time-varying voltage representation using polynomials:
```
theta(t) = theta_0 + theta_1*t + theta_2*t^2 + ...
V(t) = V_0 + V_1*t + V_2*t^2 + ...
```

Default: Quadratic profiles (6 parameters)

### 3. Sampling Strategies

**Time Sampling**:
- **Linear**: Uniform distribution
- **Log**: Logarithmic spacing for fast dynamics

**State Sampling** (around equilibrium):
- **Radial**: Linear radial distribution
- **Annular**: Uniform in annular region (better coverage)
- **Square**: Uniform in square region

**Voltage Sampling**:
- Uniform sampling within specified parameter bounds

### 4. Dataset Types

**Training Dataset**:
- 2,500 samples
- Full trajectory simulation
- For model training

**Test Dataset**:
- 4,000 samples
- Full trajectory simulation
- For model evaluation

**Collocation Dataset**:
- 10,000 samples
- No simulation (only initial conditions)
- For physics-informed constraints

## Configuration

### DatasetConfig

```python
config = DatasetConfig(
    n_samples=1000,              # Number of samples
    t_min=0.0,                   # Minimum time
    t_max=0.5,                   # Maximum time
    time_sampling="linear",       # "linear" or "log"
    perturbation_radius=0.25,    # State perturbation magnitude
    state_sampling="annular",     # "radial", "annular", "square"
    voltage_order=2,             # Polynomial order
    simulate=True,               # Simulate trajectories
    rtol=1e-6,                   # ODE solver relative tolerance
    atol=1e-8,                   # ODE solver absolute tolerance
    random_seed=42,              # Random seed
    use_torch=True,              # Use PyTorch tensors
    backend="scipy"              # "scipy" or "torch"
)
```

### Pre-configured Datasets

```python
# Training dataset
config = DatasetConfig.training_config(n_samples=2500, random_seed=42)

# Test dataset
config = DatasetConfig.test_config(n_samples=4000, random_seed=123)

# Collocation dataset
config = DatasetConfig.collocation_config(n_samples=10000, random_seed=456)
```

### Generator Parameters

**IEEE 9-bus generators**:
```python
from power_system_datagen.config import get_generator_parameters

# Get predefined parameters
params = get_generator_parameters("ieee9_1")  # or "ieee9_2", "ieee9_3"
```

**Custom generator**:
```python
from power_system_datagen.config import create_custom_generator

params = create_custom_generator(
    H=5.0,              # Inertia constant
    D=2.0,              # Damping
    X_d=1.0,            # d-axis reactance
    X_q=0.6,            # q-axis reactance
    X_d_prime=0.3,      # Transient reactances
    X_q_prime=0.3,
    T_d0_prime=6.0,     # Time constants
    T_q0_prime=0.5,
    set_point=[1.0, 0.2, 0.0, 1.0]  # [P, Q, theta, V]
)
```

## Dataset Structure

Generated datasets contain:

```python
dataset = {
    # Input data
    "time": (n_samples, 1),                    # Time values
    "state_initial": (n_samples, 4),           # Initial states
    "control_input": (n_samples, 2),           # Control inputs
    "voltage_parametrisation": (n_samples, 6), # Voltage parameters
    "state_equilibrium": (n_samples, 4),       # Equilibrium states

    # Output data (if simulated)
    "state_result": (n_samples, 4),            # Final states
    "theta_result": (n_samples, 1),            # Voltage angles
    "V_result": (n_samples, 1),                # Voltage magnitudes

    # Metadata
    "generator_params": {...},                 # Generator parameters
    "config": {...}                            # Configuration
}
```

## Examples

See the `examples/` directory for complete examples:

1. **basic_usage.py**: Simple dataset generation with visualization
2. **generate_all_datasets.py**: Generate all standard datasets for IEEE 9-bus
3. **custom_generator.py**: Using custom generator parameters

### Run Examples

```bash
cd examples
python basic_usage.py
python generate_all_datasets.py
python custom_generator.py
```

## Visualization

### Plot Trajectories

```python
from power_system_datagen.utils import plot_trajectories

fig = plot_trajectories(
    dataset,
    n_trajectories=10,
    state_names=["E_q'", "E_d'", "delta", "omega"]
)
```

### Plot Statistics

```python
from power_system_datagen.utils import plot_dataset_statistics

fig = plot_dataset_statistics(
    dataset,
    state_names=["E_q'", "E_d'", "delta", "omega"]
)
```

### Phase Portrait

```python
from power_system_datagen.utils import plot_phase_portrait

fig = plot_phase_portrait(
    dataset,
    state_idx1=2,  # delta
    state_idx2=3,  # omega
)
```

## Data I/O

### Save Dataset

```python
from power_system_datagen.utils import save_dataset

# Pickle format (recommended)
save_dataset(dataset, "my_data.pkl", format="pickle")

# NumPy format
save_dataset(dataset, "my_data.npz", format="npz")
```

### Load Dataset

```python
from power_system_datagen.utils import load_dataset

# Load as NumPy arrays
dataset = load_dataset("my_data.pkl", format="pickle")

# Load as PyTorch tensors
dataset = load_dataset("my_data.pkl", format="pickle", use_torch=True)
```

## Technical Details

### Generator Dynamics

The framework implements a 4th-order synchronous generator model based on:

**Electromagnetic dynamics** (d-q reference frame):
```
T_d0' * dE_q'/dt = -E_q' + (X_d - X_d')*I_d + E_fd
T_q0' * dE_d'/dt = -E_d' - (X_q - X_q')*I_q
```

**Mechanical dynamics** (swing equation):
```
2H * d(omega)/dt = P_M - P_e - D*omega
d(delta)/dt = omega * omega_s
```

Where:
- Fast electromagnetic transients are neglected (dE_q'/dt ≈ 0, dE_d'/dt ≈ 0)
- Electrical power: `P_e = E_q'*I_q + E_d'*I_d + (X_q' - X_d')*I_d*I_q`
- Classical model assumption: `X_q = X_q' = X_d'`

### ODE Solver

Uses SciPy's `solve_ivp` with RK45 method (adaptive Runge-Kutta):
- High accuracy (default: rtol=1e-6, atol=1e-8)
- Automatic step size control
- Efficient for stiff and non-stiff problems

## Applications

This framework is designed for:

1. **Physics-Informed Neural Networks (PINNs)**: Generate collocation points
2. **Surrogate Modeling**: Train fast approximations of generator dynamics
3. **State Estimation**: Training data for estimators
4. **Stability Analysis**: Explore transient behavior under various conditions
5. **Control Design**: Data-driven controller development

## References

- P. Kundur, "Power System Stability and Control", McGraw-Hill, 1994
- P. Sauer, M. Pai, "Power System Dynamics and Stability", Prentice Hall, 1998
- IEEE 9-bus test system parameters

## License

This framework is provided as-is for research and educational purposes.

## Support

For issues, questions, or contributions, please refer to the project documentation.

---

**Version**: 1.0.0
**Last Updated**: 2025
