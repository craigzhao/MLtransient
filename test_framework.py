"""
Test script for the power system data generation framework.

Runs basic tests to verify all components work correctly.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, 'power_system_datagen')

from power_system_datagen import DataGenerator
from power_system_datagen.config import DatasetConfig, get_generator_parameters
from power_system_datagen.models import SynchronousGenerator, VoltageProfilePolynomial
from power_system_datagen.sampling import sample_time, sample_initial_states, sample_voltage_parameters
from power_system_datagen.simulation import ODESimulator
from power_system_datagen.utils import save_dataset, load_dataset


def test_generator_model():
    """Test synchronous generator model."""
    print("\n[TEST 1] Testing Generator Model...")

    params = get_generator_parameters("ieee9_1")
    gen = SynchronousGenerator(params)

    # Test equilibrium computation
    set_point = np.array([[0.716, 0.27, 0.0, 1.04]])
    eq_state, eq_control = gen.compute_equilibrium(set_point)

    assert eq_state.shape == (1, 4), f"Wrong state shape: {eq_state.shape}"
    assert eq_control.shape == (1, 2), f"Wrong control shape: {eq_control.shape}"

    # Test dynamics
    state = eq_state
    control = eq_control
    theta = set_point[:, 2:3]
    V = set_point[:, 3:4]

    dstate = gen.dynamics(0.0, state, control, theta, V)
    assert dstate.shape == (1, 4), f"Wrong derivative shape: {dstate.shape}"

    # At equilibrium, omega should be zero and ddelta/dt should be near zero
    assert np.abs(dstate[0, 2]) < 1e-6, f"delta derivative not near zero: {dstate[0, 2]}"
    assert np.abs(dstate[0, 3]) < 1e-3, f"omega derivative not near zero: {dstate[0, 3]}"

    print("    ✓ Generator model tests passed")


def test_voltage_profile():
    """Test voltage profile model."""
    print("\n[TEST 2] Testing Voltage Profile...")

    vp = VoltageProfilePolynomial(order=2)
    assert vp.n_parameters == 6, f"Wrong number of parameters: {vp.n_parameters}"

    # Test evaluation
    time = np.array([[0.1], [0.2], [0.3]])
    params = np.array([[0.0, 1.0, 0.1, 0.0, 0.0, 0.0]]).repeat(3, axis=0)

    theta, V = vp.evaluate(time, params)
    assert theta.shape == (3, 1), f"Wrong theta shape: {theta.shape}"
    assert V.shape == (3, 1), f"Wrong V shape: {V.shape}"

    # Check values
    expected_V = np.array([[1.0], [1.0], [1.0]])
    assert np.allclose(V, expected_V), f"Voltage values incorrect"

    print("    ✓ Voltage profile tests passed")


def test_sampling():
    """Test sampling functions."""
    print("\n[TEST 3] Testing Sampling Functions...")

    # Test time sampling
    time_samples = sample_time(100, t_min=0.0, t_max=1.0, sampling_mode="linear", use_torch=False)
    assert time_samples.shape == (100, 1), f"Wrong time shape: {time_samples.shape}"
    assert np.min(time_samples) >= 0.0 and np.max(time_samples) <= 1.0, "Time out of bounds"

    # Test state sampling
    eq_state = np.array([[1.0, 0.0, 0.5, 0.0]])
    state_samples = sample_initial_states(
        50, eq_state, perturbation_radius=0.1, sampling_mode="annular",
        state_indices=(2, 3), use_torch=False
    )
    assert state_samples.shape == (50, 4), f"Wrong state shape: {state_samples.shape}"

    # Test voltage sampling
    param_min = np.array([-1.0, 0.9, -0.1, -0.1, -0.1, -0.1])
    param_max = np.array([1.0, 1.1, 0.1, 0.1, 0.1, 0.1])
    voltage_samples = sample_voltage_parameters(
        50, param_min, param_max, use_torch=False
    )
    assert voltage_samples.shape == (50, 6), f"Wrong voltage param shape: {voltage_samples.shape}"

    print("    ✓ Sampling tests passed")


def test_simulation():
    """Test ODE simulator."""
    print("\n[TEST 4] Testing ODE Simulator...")

    params = get_generator_parameters("ieee9_1")
    gen = SynchronousGenerator(params)
    vp = VoltageProfilePolynomial(order=2)
    sim = ODESimulator(gen, vp, backend="scipy")

    # Setup initial condition
    set_point = np.array([[0.716, 0.27, 0.0, 1.04]])
    eq_state, eq_control = gen.compute_equilibrium(set_point)

    # Perturb slightly
    x0 = eq_state.flatten() + np.array([0.0, 0.0, 0.1, 0.01])
    u = eq_control.flatten()
    v_params = np.array([0.0, 1.04, 0.0, 0.0, 0.0, 0.0])

    # Simulate
    time_points = np.array([0.0, 0.1])
    trajectory = sim.simulate(time_points, x0, u, v_params)

    assert trajectory.shape == (2, 4), f"Wrong trajectory shape: {trajectory.shape}"
    assert np.allclose(trajectory[0], x0, atol=1e-6), "Initial state mismatch"

    print("    ✓ Simulation tests passed")


def test_data_generation():
    """Test complete data generation pipeline."""
    print("\n[TEST 5] Testing Data Generation Pipeline...")

    config = DatasetConfig(
        n_samples=10,
        t_min=0.0,
        t_max=0.1,
        simulate=True,
        random_seed=42,
        use_torch=False
    )

    gen = DataGenerator("ieee9_1", config)
    dataset = gen.generate_dataset(verbose=False)

    # Check dataset structure
    assert "time" in dataset, "Missing 'time' in dataset"
    assert "state_initial" in dataset, "Missing 'state_initial' in dataset"
    assert "state_result" in dataset, "Missing 'state_result' in dataset"
    assert "control_input" in dataset, "Missing 'control_input' in dataset"
    assert "voltage_parametrisation" in dataset, "Missing 'voltage_parametrisation' in dataset"

    # Check shapes
    assert dataset["time"].shape == (10, 1), f"Wrong time shape: {dataset['time'].shape}"
    assert dataset["state_initial"].shape == (10, 4), f"Wrong initial state shape"
    assert dataset["state_result"].shape == (10, 4), f"Wrong result state shape"

    print("    ✓ Data generation tests passed")


def test_io():
    """Test data I/O."""
    print("\n[TEST 6] Testing Data I/O...")

    # Create small dataset
    config = DatasetConfig(n_samples=5, simulate=False, use_torch=False)
    gen = DataGenerator("ieee9_1", config)
    dataset = gen.generate_dataset(verbose=False)

    # Save and load
    filename = "power_system_datagen/data/test_dataset.pkl"
    save_dataset(dataset, filename, format="pickle")
    loaded = load_dataset(filename, format="pickle", use_torch=False)

    # Verify
    assert "time" in loaded, "Missing 'time' after loading"
    assert loaded["time"].shape == dataset["time"].shape, "Shape mismatch after loading"

    print("    ✓ I/O tests passed")


def main():
    print("=" * 70)
    print("Power System Data Generation Framework - Test Suite")
    print("=" * 70)

    try:
        test_generator_model()
        test_voltage_profile()
        test_sampling()
        test_simulation()
        test_data_generation()
        test_io()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
