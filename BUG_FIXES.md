# Bug Fixes Applied to Power System Data Generation Framework

## Summary of Bugs Fixed

### **LATEST FIX (11)**: List Input Handling in Simulator
- **File**: `simulation/simulator.py`
- **Problem**: `time_points` passed as list caused AttributeError (lists don't have `.shape`)
- **Impact**: `AttributeError: 'list' object has no attribute 'shape'` during simulation
- **Fix**: Check for list type first and convert to numpy/torch before checking shape
- **Lines**: 67-70, 127-145 in `simulation/simulator.py`

### **FIX 9**: Type Mismatch in sample_initial_states
- **File**: `sampling/samplers.py`
- **Problem**: `equilibrium_state` type didn't match `use_torch` setting
- **Impact**: TypeError when adding equilibrium_state to perturbations
- **Fix**: Convert equilibrium_state to torch.Tensor when `use_torch=True`, to numpy array when `use_torch=False`
- **Lines**: 112-113, 157-160

### 1. **Missing Import in config/__init__.py**
- **Bug**: `create_custom_generator` was defined but not exported
- **Impact**: Custom generator example would fail with import error
- **Fix**: Added `create_custom_generator` to exports in `config/__init__.py`

### 2. **Complex Number Handling in generator_model.py**
- **Bug**: `.real` and `.imag` properties on numpy complex arrays return complex dtype, not float
- **Impact**: State calculations would have wrong data type, causing errors
- **Fix**: Wrapped with `np.real()` and `torch.real()` to explicitly convert to real numbers
- **Location**: Lines 157-160, 174-177 in `models/generator_model.py`

### 3. **Matrix Type Mismatch in generator_model.py**
- **Bug**: `Z_inv` stored as numpy array, incompatible with PyTorch tensors
- **Impact**: Would fail when using torch tensors
- **Fix**: Store inverse matrix elements as float scalars for compatibility
- **Location**: Lines 64-73 in `models/generator_model.py`

### 4. **Mixed Numpy/Torch in generator.py**
- **Bug**: Used `np.zeros` for base_voltage_params even when `use_torch=True`
- **Impact**: Type mismatch errors during voltage parameter sampling
- **Fix**: Create torch.zeros when use_torch=True, np.zeros otherwise
- **Location**: Lines 141-146 in `generator.py`

### 5. **Incorrect Matrix Multiplication in generator.py**
- **Bug**: Matrix multiplication `theta_0 @ tensor([[0., 0., 1., 0.]]).T` had wrong dimensions
- **Impact**: Broadcasting error when adjusting initial delta states
- **Fix**: Create zero array same shape as initial_states, then assign theta_0 to delta column
- **Location**: Lines 158-167 in `generator.py`

### 6. **Tensor Conversion Error in generator.py**
- **Bug**: `torch.tensor()` called on already-tensor data (control_input, equilibrium_state)
- **Impact**: Warning or error when data is already a tensor
- **Fix**: Check if already tensor before converting
- **Location**: Lines 171-172 in `generator.py`

### 7. **Mixed Numpy/Torch Constants in samplers.py**
- **Bug**: Used `np.pi` and `np.sqrt(np.pi)` with torch tensors
- **Impact**: Creates mixed numpy/torch tensors, causing computation errors
- **Fix**: Convert to `torch.tensor(np.pi)` and use `torch.sqrt()` when use_torch=True
- **Location**: Lines 78-100 in `sampling/samplers.py`

### 8. **Missing Exports in utils/__init__.py**
- **Bug**: `print_dataset_summary` and `plot_phase_portrait` not exported
- **Impact**: Import errors in example scripts
- **Fix**: Added both functions to `__all__` list in `utils/__init__.py`

### **FIX 10**: Missing Export of set_random_seed
- **File**: `sampling/__init__.py`
- **Problem**: `set_random_seed` function not exported from sampling module
- **Impact**: ImportError when trying to import in generator.py
- **Fix**: Added `set_random_seed` to exports in `sampling/__init__.py`

## Files Modified

1. `power_system_datagen/config/__init__.py`
2. `power_system_datagen/models/generator_model.py`
3. `power_system_datagen/generator.py`
4. `power_system_datagen/sampling/__init__.py`
5. `power_system_datagen/sampling/samplers.py`
6. `power_system_datagen/simulation/simulator.py`
7. `power_system_datagen/utils/__init__.py`

## Testing Recommendations

Before using the framework, test with:

```python
# Test 1: Import all modules
from power_system_datagen import DataGenerator
from power_system_datagen.config import DatasetConfig, create_custom_generator
from power_system_datagen.utils import print_dataset_summary, plot_phase_portrait

# Test 2: Create simple dataset
config = DatasetConfig(n_samples=10, simulate=False, use_torch=False)
gen = DataGenerator("ieee9_1", config)
dataset = gen.generate_dataset(verbose=True)

# Test 3: Verify equilibrium computation
import numpy as np
set_point = np.array([[0.716, 0.27, 0.0, 1.04]])
eq_state, eq_control = gen.generator.compute_equilibrium(set_point)
print(f"Equilibrium state shape: {eq_state.shape}")
print(f"Equilibrium control shape: {eq_control.shape}")
```

## Potential Remaining Issues

1. **Simulation backend**: The torch backend for ODE solver needs `torchdiffeq` package
2. **Visualization**: Matplotlib may have compatibility issues with some environments
3. **Large datasets**: Memory management for very large datasets (>100k samples) may need optimization

## Bug Categories Fixed

- ✅ Import/Export errors (3 bugs: #1, #8, #10)
- ✅ Type mismatches (numpy vs torch) (6 bugs: #3, #4, #6, #7, #9, #11)
- ✅ Complex number handling (1 bug: #2)
- ✅ Matrix dimension errors (1 bug: #5)

**Total bugs fixed: 11**

All major bugs have been identified and fixed. The framework should now work correctly for both numpy and PyTorch backends.
