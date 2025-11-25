"""
Dataset configuration class.

Defines all parameters for dataset generation.
"""

import numpy as np


class DatasetConfig:
    """
    Configuration for dataset generation.

    Specifies sampling strategies and parameters for all aspects
    of data generation.
    """

    def __init__(
        self,
        # Dataset size
        n_samples=1000,
        # Time sampling
        t_min=0.0,
        t_max=0.5,
        time_sampling="linear",  # "linear" or "log"
        # State sampling
        perturbation_radius=0.25,
        state_sampling="annular",  # "radial", "annular", or "square"
        perturbed_states=(2, 3),  # Which states to perturb (delta, omega)
        # Voltage sampling
        voltage_param_min=None,
        voltage_param_max=None,
        voltage_order=2,
        # Simulation
        simulate=True,
        rtol=1e-6,
        atol=1e-8,
        # Other
        random_seed=None,
        use_torch=True,
        backend="scipy",
    ):
        """
        Initialize dataset configuration.

        Args:
            n_samples: Number of data samples
            t_min: Minimum time value
            t_max: Maximum time value
            time_sampling: Time sampling mode ("linear" or "log")
            perturbation_radius: State perturbation magnitude
            state_sampling: State sampling mode ("radial", "annular", "square")
            perturbed_states: Tuple of state indices to perturb
            voltage_param_min: Minimum voltage parameters (array-like)
            voltage_param_max: Maximum voltage parameters (array-like)
            voltage_order: Polynomial order for voltage profile
            simulate: Whether to simulate trajectories (False for collocation data)
            rtol: ODE solver relative tolerance
            atol: ODE solver absolute tolerance
            random_seed: Random seed for reproducibility
            use_torch: Use PyTorch tensors
            backend: Simulation backend ("scipy" or "torch")
        """
        self.n_samples = n_samples
        self.t_min = t_min
        self.t_max = t_max
        self.time_sampling = time_sampling
        self.perturbation_radius = perturbation_radius
        self.state_sampling = state_sampling
        self.perturbed_states = perturbed_states
        self.voltage_order = voltage_order
        self.simulate = simulate
        self.rtol = rtol
        self.atol = atol
        self.random_seed = random_seed
        self.use_torch = use_torch
        self.backend = backend

        # Default voltage parameter ranges (for order=2, 6 parameters)
        if voltage_param_min is None:
            self.voltage_param_min = np.array([
                -np.pi, -0.1, -0.3, -0.4, -0.8, -0.5
            ])
        else:
            self.voltage_param_min = np.array(voltage_param_min)

        if voltage_param_max is None:
            self.voltage_param_max = np.array([
                np.pi, 0.35, 0.3, 0.4, 0.8, 0.5
            ])
        else:
            self.voltage_param_max = np.array(voltage_param_max)

    @classmethod
    def training_config(cls, n_samples=2500, random_seed=None):
        """Create configuration for training dataset."""
        return cls(
            n_samples=n_samples,
            simulate=True,
            time_sampling="linear",
            state_sampling="annular",
            random_seed=random_seed,
        )

    @classmethod
    def test_config(cls, n_samples=4000, random_seed=None):
        """Create configuration for test dataset."""
        return cls(
            n_samples=n_samples,
            simulate=True,
            time_sampling="linear",
            state_sampling="annular",
            random_seed=random_seed,
        )

    @classmethod
    def collocation_config(cls, n_samples=10000, random_seed=None):
        """Create configuration for collocation dataset (no simulation)."""
        return cls(
            n_samples=n_samples,
            simulate=False,
            time_sampling="linear",
            state_sampling="annular",
            random_seed=random_seed,
        )

    def __repr__(self):
        return (f"DatasetConfig(n_samples={self.n_samples}, "
                f"time={self.time_sampling}, state={self.state_sampling}, "
                f"simulate={self.simulate})")
