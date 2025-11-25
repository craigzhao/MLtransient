# Quick Start Guide

## Installation

1. **Install dependencies**:
```bash
pip install numpy scipy torch matplotlib
```

2. **Verify installation**:
```bash
python -c "import numpy, scipy, torch, matplotlib; print('All dependencies installed!')"
```

## Running Your First Example

### Option 1: Basic Example (Recommended)

```bash
cd power_system_datagen/examples
python basic_usage.py
```

This will:
- Generate a dataset with 500 samples
- Simulate generator trajectories
- Create visualizations
- Save dataset to `../data/example_dataset.pkl`

### Option 2: Generate Standard Datasets

```bash
cd power_system_datagen/examples
python generate_all_datasets.py
```

This generates train/test/collocation datasets for all three IEEE 9-bus generators.

### Option 3: Custom Generator

```bash
cd power_system_datagen/examples
python custom_generator.py
```

Shows how to define your own generator parameters.

## Python Script Example

```python
#!/usr/bin/env python3
"""
Minimal example - generates a small dataset
"""

import sys
sys.path.append('power_system_datagen')

from power_system_datagen import DataGenerator
from power_system_datagen.config import DatasetConfig

# Configure dataset
config = DatasetConfig(
    n_samples=100,          # Generate 100 samples
    t_max=0.5,             # Simulate up to 0.5 seconds
    simulate=True,          # Run ODE simulation
    random_seed=42          # For reproducibility
)

# Create generator using IEEE 9-bus Gen 1 parameters
generator = DataGenerator("ieee9_1", config)

# Generate dataset
print("Generating dataset...")
dataset = generator.generate_dataset(verbose=True)

# Save
generator.save(dataset, "my_first_dataset.pkl")
print(f"Dataset saved!")
print(f"Time shape: {dataset['time'].shape}")
print(f"States shape: {dataset['state_result'].shape}")
```

## Understanding the Output

Each dataset contains:

```
dataset = {
    'time': Time values for each sample [n_samples, 1]
    'state_initial': Initial states [n_samples, 4]
    'state_result': Final states after simulation [n_samples, 4]
    'control_input': Control inputs [n_samples, 2]
    'voltage_parametrisation': Voltage parameters [n_samples, 6]
    'theta_result': Voltage angles [n_samples, 1]
    'V_result': Voltage magnitudes [n_samples, 1]
}
```

States are: `[E_q', E_d', delta, omega]`
Controls are: `[P_M, E_fd]`

## Troubleshooting

**Import Error: No module named 'power_system_datagen'**
- Make sure you're running from the correct directory
- Use `sys.path.append('path/to/power_system_datagen')` before importing

**Import Error: No module named 'numpy'**
- Install dependencies: `pip install numpy scipy torch matplotlib`

**Simulation takes too long**
- Reduce `n_samples` in configuration
- Use `simulate=False` for collocation data (faster)

**Memory issues**
- Generate datasets in smaller batches
- Use `use_torch=False` to avoid GPU memory

## Next Steps

1. **Explore visualizations**: Check the `utils/visualization.py` module
2. **Customize parameters**: See `config/generator_params.py` for generator configs
3. **Modify sampling**: Adjust sampling strategies in `config/dataset_config.py`
4. **Use the data**: Load datasets with `load_dataset()` for ML training

## Tips

- Start with small datasets (`n_samples=100`) to test
- Use `random_seed` for reproducible results
- Set `verbose=True` to see progress
- Check the `examples/` directory for more use cases
