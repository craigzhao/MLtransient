"""
Main data generation orchestrator.

Combines all components to generate physics-based training datasets.
"""

import time
import numpy as np
import torch

from .models import SynchronousGenerator, VoltageProfilePolynomial
from .sampling import sample_time, sample_initial_states, sample_voltage_parameters, set_random_seed
from .simulation import ODESimulator
from .config import DatasetConfig, get_generator_parameters
from .utils import save_dataset


class DataGenerator:
    """
    Physics-based data generator for power system dynamics.

    Generates datasets by:
    1. Sampling operating conditions (time, states, voltage profiles)
    2. Simulating generator dynamics using ODE solver
    3. Storing trajectory data for machine learning
    """

    def __init__(self, generator_params, dataset_config):
        """
        Initialize data generator.

        Args:
            generator_params: Generator parameter dictionary or ID string
            dataset_config: DatasetConfig instance
        """
        # Load generator parameters
        if isinstance(generator_params, str):
            generator_params = get_generator_parameters(generator_params)

        self.generator_params = generator_params
        self.config = dataset_config

        # Initialize models
        self.generator = SynchronousGenerator(generator_params)
        self.voltage_profile = VoltageProfilePolynomial(order=dataset_config.voltage_order)

        # Initialize simulator
        self.simulator = ODESimulator(
            self.generator,
            self.voltage_profile,
            backend=dataset_config.backend
        )

        # Compute equilibrium point
        set_point = np.array(generator_params["set_point"]).reshape(1, -1)
        self.equilibrium_state, self.equilibrium_control = self.generator.compute_equilibrium(set_point)

        # Apply control adjustments if specified
        control_adjustment = generator_params.get("control_adjustment", [1.0, 1.0])
        self.control_input = self.equilibrium_control * np.array(control_adjustment).reshape(1, -1)

        # Get set point voltage for base voltage parameters
        self.set_point_voltage = set_point[0, 3]

    def generate_dataset(self, verbose=True):
        """
        Generate complete dataset.

        Returns:
            Dictionary containing all dataset components
        """
        start_time = time.time()

        # Set random seed if specified
        if self.config.random_seed is not None:
            set_random_seed(self.config.random_seed)

        if verbose:
            print(f"Generating dataset with {self.config.n_samples} samples...")
            print(f"Configuration: {self.config}")

        # Sample data
        dataset = self._sample_dataset(verbose=verbose)

        # Simulate trajectories if requested
        if self.config.simulate:
            if verbose:
                print("Simulating trajectories...")
            dataset = self._simulate_dataset(dataset, verbose=verbose)
        else:
            if verbose:
                print("Skipping simulation (collocation mode)")
            # Just compute voltage values
            dataset = self._add_voltage_values(dataset)

        # Add metadata
        dataset["generator_params"] = self.generator_params
        dataset["config"] = {
            "n_samples": self.config.n_samples,
            "time_sampling": self.config.time_sampling,
            "state_sampling": self.config.state_sampling,
            "simulated": self.config.simulate,
        }

        elapsed = time.time() - start_time
        if verbose:
            print(f"Dataset generation completed in {elapsed:.2f} seconds")

        return dataset

    def _sample_dataset(self, verbose=False):
        """Sample all dataset components."""
        n = self.config.n_samples

        # Sample time values
        time_samples = sample_time(
            n,
            t_min=self.config.t_min,
            t_max=self.config.t_max,
            sampling_mode=self.config.time_sampling,
            use_torch=self.config.use_torch
        )

        # Sample initial states (perturbed from equilibrium)
        scaling = [
            self.generator.norm_to_scale_delta,
            self.generator.norm_to_scale_omega
        ]
        initial_states = sample_initial_states(
            n,
            equilibrium_state=self.equilibrium_state,
            perturbation_radius=self.config.perturbation_radius,
            sampling_mode=self.config.state_sampling,
            state_indices=self.config.perturbed_states,
            scaling_factors=scaling,
            use_torch=self.config.use_torch
        )

        # Sample voltage parameters
        # Base parameters on equilibrium voltage
        base_voltage_params = np.zeros((1, self.voltage_profile.n_parameters))
        base_voltage_params[0, 1] = self.set_point_voltage  # V_0 term

        voltage_params = sample_voltage_parameters(
            n,
            param_min=self.config.voltage_param_min,
            param_max=self.config.voltage_param_max,
            base_value=base_voltage_params,
            use_torch=self.config.use_torch
        )

        # Adjust initial delta based on initial voltage angle
        theta_0, _ = self.voltage_profile.get_initial_voltage(voltage_params)
        if self.config.use_torch:
            delta_adjustment = theta_0 @ torch.tensor([[0., 0., 1., 0.]]).T
            initial_states = initial_states + delta_adjustment.T
        else:
            delta_adjustment = theta_0 @ np.array([[0., 0., 1., 0.]]).T
            initial_states = initial_states + delta_adjustment.T

        # Create control inputs (constant for all samples)
        if self.config.use_torch:
            control_inputs = torch.tensor(self.control_input).repeat(n, 1)
            equilibrium_states = torch.tensor(self.equilibrium_state).repeat(n, 1)
        else:
            control_inputs = np.repeat(self.control_input, n, axis=0)
            equilibrium_states = np.repeat(self.equilibrium_state, n, axis=0)

        dataset = {
            "time": time_samples,
            "state_initial": initial_states,
            "control_input": control_inputs,
            "state_equilibrium": equilibrium_states,
            "voltage_parametrisation": voltage_params,
        }

        if verbose:
            print(f"  Sampled {n} operating conditions")

        return dataset

    def _simulate_dataset(self, dataset, verbose=False):
        """Simulate trajectories for dataset."""
        state_results = self.simulator.simulate_batch(
            dataset["time"],
            dataset["state_initial"],
            dataset["control_input"],
            dataset["voltage_parametrisation"],
            rtol=self.config.rtol,
            atol=self.config.atol,
            verbose=verbose
        )

        dataset["state_result"] = state_results

        # Add voltage values at final time
        dataset = self._add_voltage_values(dataset)

        return dataset

    def _add_voltage_values(self, dataset):
        """Compute voltage angle and magnitude at sampled times."""
        theta, V = self.voltage_profile.evaluate(
            dataset["time"],
            dataset["voltage_parametrisation"]
        )

        dataset["theta_result"] = theta
        dataset["V_result"] = V

        return dataset

    def save(self, dataset, filename, format="pickle"):
        """
        Save generated dataset.

        Args:
            dataset: Dataset dictionary
            filename: Output filename
            format: File format ("pickle" or "npz")
        """
        save_dataset(dataset, filename, format=format)


def generate_standard_datasets(generator_id="ieee9_1", output_dir="./data", verbose=True):
    """
    Generate standard train/test/collocation datasets.

    Args:
        generator_id: Generator identifier
        output_dir: Output directory for datasets
        verbose: Print progress

    Returns:
        Dictionary with dataset names and paths
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = {}

    # Training dataset
    if verbose:
        print("\n" + "=" * 60)
        print("Generating TRAINING dataset")
        print("=" * 60)

    train_config = DatasetConfig.training_config(n_samples=2500, random_seed=42)
    train_gen = DataGenerator(generator_id, train_config)
    train_data = train_gen.generate_dataset(verbose=verbose)
    train_file = output_path / f"train_{generator_id}.pkl"
    train_gen.save(train_data, train_file)
    datasets["train"] = train_file

    # Test dataset
    if verbose:
        print("\n" + "=" * 60)
        print("Generating TEST dataset")
        print("=" * 60)

    test_config = DatasetConfig.test_config(n_samples=4000, random_seed=123)
    test_gen = DataGenerator(generator_id, test_config)
    test_data = test_gen.generate_dataset(verbose=verbose)
    test_file = output_path / f"test_{generator_id}.pkl"
    test_gen.save(test_data, test_file)
    datasets["test"] = test_file

    # Collocation dataset
    if verbose:
        print("\n" + "=" * 60)
        print("Generating COLLOCATION dataset")
        print("=" * 60)

    colloc_config = DatasetConfig.collocation_config(n_samples=10000, random_seed=456)
    colloc_gen = DataGenerator(generator_id, colloc_config)
    colloc_data = colloc_gen.generate_dataset(verbose=verbose)
    colloc_file = output_path / f"collocation_{generator_id}.pkl"
    colloc_gen.save(colloc_data, colloc_file)
    datasets["collocation"] = colloc_file

    if verbose:
        print("\n" + "=" * 60)
        print("All datasets generated successfully!")
        print("=" * 60)

    return datasets
