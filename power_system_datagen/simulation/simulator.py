"""
ODE Simulator for power system dynamics.

Integrates generator equations with time-varying voltage profiles.
"""

import torch
import numpy as np
from scipy.integrate import solve_ivp


class ODESimulator:
    """
    ODE Simulator for synchronous generator with voltage profile.

    Supports both PyTorch (with torchdiffeq) and NumPy/SciPy backends.
    """

    def __init__(self, generator_model, voltage_profile, backend="scipy"):
        """
        Initialize ODE simulator.

        Args:
            generator_model: SynchronousGenerator instance
            voltage_profile: VoltageProfilePolynomial instance
            backend: "scipy" or "torch" (requires torchdiffeq)
        """
        self.generator = generator_model
        self.voltage_profile = voltage_profile
        self.backend = backend

        if backend == "torch":
            try:
                import torchdiffeq
                self.torchdiffeq = torchdiffeq
            except ImportError:
                raise ImportError("torchdiffeq is required for PyTorch backend. "
                                  "Install with: pip install torchdiffeq")

    def simulate(self, time_points, initial_state, control_input, voltage_params,
                 rtol=1e-6, atol=1e-8):
        """
        Simulate generator dynamics.

        Args:
            time_points: Time points to evaluate [t0, t1, ..., tn]
            initial_state: Initial state vector
            control_input: Control input vector (constant)
            voltage_params: Voltage profile parameters
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            state_trajectory: State values at time_points
        """
        if self.backend == "torch":
            return self._simulate_torch(time_points, initial_state, control_input,
                                         voltage_params, rtol, atol)
        else:
            return self._simulate_scipy(time_points, initial_state, control_input,
                                         voltage_params, rtol, atol)

    def _simulate_torch(self, time_points, initial_state, control_input,
                        voltage_params, rtol, atol):
        """Simulate using PyTorch and torchdiffeq."""
        # Ensure tensors
        if isinstance(time_points, list):
            time_points = torch.tensor(time_points, dtype=torch.float32)
        elif not torch.is_tensor(time_points):
            time_points = torch.tensor(time_points, dtype=torch.float32)

        if not torch.is_tensor(initial_state):
            initial_state = torch.tensor(initial_state, dtype=torch.float32)

        if not torch.is_tensor(control_input):
            control_input = torch.tensor(control_input, dtype=torch.float32)

        if not torch.is_tensor(voltage_params):
            voltage_params = torch.tensor(voltage_params, dtype=torch.float32)

        # Flatten for ODE solver
        if len(time_points.shape) > 1:
            time_points = time_points.flatten()
        if len(initial_state.shape) > 1:
            initial_state = initial_state.flatten()
        if len(control_input.shape) > 1:
            control_input = control_input.flatten()
        if len(voltage_params.shape) > 1:
            voltage_params = voltage_params.flatten()

        # Store control and voltage params for dynamics function
        self._control_input = control_input
        self._voltage_params = voltage_params

        def dynamics_func(t, state):
            """ODE function for solver."""
            # Evaluate voltage at time t
            t_tensor = torch.tensor([[t]], dtype=torch.float32)
            theta, V = self.voltage_profile.evaluate(t_tensor, self._voltage_params.reshape(1, -1))
            theta = theta.item()
            V = V.item()

            # Compute derivatives
            state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
            control_tensor = self._control_input.reshape(1, -1)
            dstate = self.generator.dynamics(
                t, state_tensor,
                control_tensor,
                torch.tensor([[theta]]),
                torch.tensor([[V]])
            )
            return dstate.flatten().numpy()

        # Solve ODE using scipy (more stable for torchdiffeq edge cases)
        solution = solve_ivp(
            dynamics_func,
            (time_points[0].item(), time_points[-1].item()),
            initial_state.numpy(),
            t_eval=time_points.numpy(),
            rtol=rtol,
            atol=atol,
            method='RK45'
        )

        # Convert back to torch
        return torch.tensor(solution.y.T, dtype=torch.float32)

    def _simulate_scipy(self, time_points, initial_state, control_input,
                        voltage_params, rtol, atol):
        """Simulate using SciPy."""
        # Convert to numpy
        if isinstance(time_points, list):
            time_points = np.array(time_points)
        elif torch.is_tensor(time_points):
            time_points = time_points.numpy()

        if torch.is_tensor(initial_state):
            initial_state = initial_state.numpy()
        elif not isinstance(initial_state, np.ndarray):
            initial_state = np.array(initial_state)

        if torch.is_tensor(control_input):
            control_input = control_input.numpy()
        elif not isinstance(control_input, np.ndarray):
            control_input = np.array(control_input)

        if torch.is_tensor(voltage_params):
            voltage_params = voltage_params.numpy()
        elif not isinstance(voltage_params, np.ndarray):
            voltage_params = np.array(voltage_params)

        # Flatten arrays
        if len(time_points.shape) > 1:
            time_points = time_points.flatten()
        if len(initial_state.shape) > 1:
            initial_state = initial_state.flatten()
        if len(control_input.shape) > 1:
            control_input = control_input.flatten()
        if len(voltage_params.shape) > 1:
            voltage_params = voltage_params.flatten()

        def dynamics_func(t, state):
            """ODE function for solver."""
            # Evaluate voltage at time t
            t_array = np.array([[t]])
            theta, V = self.voltage_profile.evaluate(t_array, voltage_params.reshape(1, -1))
            theta = theta.item()
            V = V.item()

            # Compute derivatives
            dstate = self.generator.dynamics(
                t,
                state.reshape(1, -1),
                control_input.reshape(1, -1),
                np.array([[theta]]),
                np.array([[V]])
            )
            return dstate.flatten()

        # Solve ODE
        solution = solve_ivp(
            dynamics_func,
            (time_points[0], time_points[-1]),
            initial_state,
            t_eval=time_points,
            rtol=rtol,
            atol=atol,
            method='RK45'
        )

        if not solution.success:
            raise RuntimeError(f"ODE integration failed: {solution.message}")

        return solution.y.T

    def simulate_batch(self, time_samples, initial_states, control_inputs,
                       voltage_params_batch, rtol=1e-6, atol=1e-8, verbose=False):
        """
        Simulate multiple trajectories in batch.

        Args:
            time_samples: Time values for each sample (shape: [n_samples, 1])
            initial_states: Initial states (shape: [n_samples, n_states])
            control_inputs: Control inputs (shape: [n_samples, n_controls])
            voltage_params_batch: Voltage parameters (shape: [n_samples, n_params])
            rtol: Relative tolerance
            atol: Absolute tolerance
            verbose: Print progress

        Returns:
            state_results: Final states at each time sample
        """
        n_samples = initial_states.shape[0]
        n_states = initial_states.shape[1]

        # Determine output type based on input type
        use_torch = torch.is_tensor(initial_states)

        if use_torch:
            state_results = torch.zeros((n_samples, n_states))
        else:
            state_results = np.zeros((n_samples, n_states))

        for i in range(n_samples):
            if verbose and (i + 1) % 100 == 0:
                print(f"Simulating trajectory {i + 1}/{n_samples}...")

            # Get sample data
            time_eval = [0.0, float(time_samples[i])]
            x0 = initial_states[i]
            u = control_inputs[i]
            v_params = voltage_params_batch[i]

            # Simulate
            trajectory = self.simulate(time_eval, x0, u, v_params, rtol, atol)

            # Store final state
            state_results[i] = trajectory[-1]

        return state_results
