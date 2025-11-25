"""
Sampling strategies for generating diverse training data.

Provides functions for sampling:
- Time values (linear or logarithmic)
- Initial states (radial, annular, or square distributions)
- Voltage profile parameters (uniform sampling)
"""

import torch
import numpy as np


def sample_time(n_samples, t_min=0.0, t_max=0.5, sampling_mode="linear", use_torch=True):
    """
    Sample time values.

    Args:
        n_samples: Number of samples
        t_min: Minimum time value
        t_max: Maximum time value
        sampling_mode: "linear" or "log"
        use_torch: Use PyTorch (True) or NumPy (False)

    Returns:
        Time samples (shape: [n_samples, 1])
    """
    if use_torch:
        if sampling_mode == "linear":
            time_values = t_min + torch.rand((n_samples, 1)) * (t_max - t_min)
        elif sampling_mode == "log":
            log_min = torch.log(torch.tensor(t_min if t_min > 0 else 1e-4))
            log_max = torch.log(torch.tensor(t_max))
            time_values = torch.exp(log_min + torch.rand((n_samples, 1)) * (log_max - log_min))
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    else:
        if sampling_mode == "linear":
            time_values = t_min + np.random.rand(n_samples, 1) * (t_max - t_min)
        elif sampling_mode == "log":
            log_min = np.log(t_min if t_min > 0 else 1e-4)
            log_max = np.log(t_max)
            time_values = np.exp(log_min + np.random.rand(n_samples, 1) * (log_max - log_min))
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    return time_values


def sample_initial_states(
    n_samples,
    equilibrium_state,
    perturbation_radius=0.25,
    sampling_mode="annular",
    state_indices=(2, 3),
    scaling_factors=None,
    use_torch=True,
):
    """
    Sample initial states around equilibrium.

    Args:
        n_samples: Number of samples
        equilibrium_state: Equilibrium state vector
        perturbation_radius: Maximum perturbation magnitude
        sampling_mode: "radial", "annular", or "square"
        state_indices: Which state indices to perturb (default: delta and omega)
        scaling_factors: Scaling for each perturbed state
        use_torch: Use PyTorch (True) or NumPy (False)

    Returns:
        Initial state samples (shape: [n_samples, n_states])
    """
    n_perturbed = len(state_indices)

    if use_torch:
        samples = torch.rand((n_samples, 2))
        pi_val = torch.tensor(np.pi)

        if sampling_mode == "radial":
            # Linear radial distribution
            radius = samples[:, 0:1] * perturbation_radius
            angle = samples[:, 1:2] * 2 * pi_val - pi_val
            perturbations = torch.cat([
                radius * torch.cos(angle),
                radius * torch.sin(angle)
            ], dim=1)

        elif sampling_mode == "annular":
            # Uniform in annular region (sqrt for uniform area)
            radius = torch.sqrt(samples[:, 0:1]) * perturbation_radius
            angle = samples[:, 1:2] * 2 * pi_val - pi_val
            perturbations = torch.cat([
                radius * torch.cos(angle),
                radius * torch.sin(angle)
            ], dim=1)

        elif sampling_mode == "square":
            # Uniform in square region
            perturbations = (samples * 2 - 1.0) * perturbation_radius / 2 * torch.sqrt(pi_val)

        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        # Apply scaling if provided
        if scaling_factors is not None:
            if isinstance(scaling_factors, (list, tuple)):
                scaling_factors = torch.tensor(scaling_factors).reshape(1, -1)
            perturbations = perturbations * scaling_factors

        # Create full state perturbations (only perturb specified indices)
        if len(equilibrium_state.shape) == 1:
            equilibrium_state = equilibrium_state.reshape(1, -1)

        n_states = equilibrium_state.shape[-1]
        full_perturbations = torch.zeros((n_samples, n_states))
        for i, idx in enumerate(state_indices):
            full_perturbations[:, idx] = perturbations[:, i]

        initial_states = equilibrium_state + full_perturbations

    else:
        samples = np.random.rand(n_samples, 2)

        if sampling_mode == "radial":
            radius = samples[:, 0:1] * perturbation_radius
            angle = samples[:, 1:2] * 2 * np.pi - np.pi
            perturbations = np.concatenate([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ], axis=1)

        elif sampling_mode == "annular":
            radius = np.sqrt(samples[:, 0:1]) * perturbation_radius
            angle = samples[:, 1:2] * 2 * np.pi - np.pi
            perturbations = np.concatenate([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ], axis=1)

        elif sampling_mode == "square":
            perturbations = (samples * 2 - 1.0) * perturbation_radius / 2 * np.sqrt(np.pi)

        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        if scaling_factors is not None:
            if isinstance(scaling_factors, (list, tuple)):
                scaling_factors = np.array(scaling_factors).reshape(1, -1)
            perturbations = perturbations * scaling_factors

        if len(equilibrium_state.shape) == 1:
            equilibrium_state = equilibrium_state.reshape(1, -1)

        n_states = equilibrium_state.shape[-1]
        full_perturbations = np.zeros((n_samples, n_states))
        for i, idx in enumerate(state_indices):
            full_perturbations[:, idx] = perturbations[:, i]

        initial_states = equilibrium_state + full_perturbations

    return initial_states


def sample_voltage_parameters(
    n_samples,
    param_min,
    param_max,
    base_value=None,
    use_torch=True,
):
    """
    Sample voltage profile parameters.

    Args:
        n_samples: Number of samples
        param_min: Minimum parameter values (array)
        param_max: Maximum parameter values (array)
        base_value: Base value to add to sampled parameters (e.g., equilibrium voltage)
        use_torch: Use PyTorch (True) or NumPy (False)

    Returns:
        Voltage parameter samples (shape: [n_samples, n_parameters])
    """
    if use_torch:
        if not torch.is_tensor(param_min):
            param_min = torch.tensor(param_min)
        if not torch.is_tensor(param_max):
            param_max = torch.tensor(param_max)

        if len(param_min.shape) == 1:
            param_min = param_min.reshape(1, -1)
        if len(param_max.shape) == 1:
            param_max = param_max.reshape(1, -1)

        n_params = param_min.shape[-1]
        samples = param_min + torch.rand((n_samples, n_params)) * (param_max - param_min)

        if base_value is not None:
            if not torch.is_tensor(base_value):
                base_value = torch.tensor(base_value)
            if len(base_value.shape) == 1:
                base_value = base_value.reshape(1, -1)
            samples = samples + base_value

    else:
        if not isinstance(param_min, np.ndarray):
            param_min = np.array(param_min)
        if not isinstance(param_max, np.ndarray):
            param_max = np.array(param_max)

        if len(param_min.shape) == 1:
            param_min = param_min.reshape(1, -1)
        if len(param_max.shape) == 1:
            param_max = param_max.reshape(1, -1)

        n_params = param_min.shape[-1]
        samples = param_min + np.random.rand(n_samples, n_params) * (param_max - param_min)

        if base_value is not None:
            if not isinstance(base_value, np.ndarray):
                base_value = np.array(base_value)
            if len(base_value.shape) == 1:
                base_value = base_value.reshape(1, -1)
            samples = samples + base_value

    return samples


def set_random_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
