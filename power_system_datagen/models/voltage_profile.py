"""
Voltage Profile Parametrization

Represents time-varying voltage using polynomial basis functions.
"""

import torch
import numpy as np


class VoltageProfilePolynomial:
    """
    Polynomial voltage profile representation.

    Voltage angle and magnitude are represented as:
        theta(t) = theta_0 + theta_1*t + theta_2*t^2 + ...
        V(t) = V_0 + V_1*t + V_2*t^2 + ...

    Parameters are interleaved: [theta_0, V_0, theta_1, V_1, theta_2, V_2, ...]
    """

    def __init__(self, order=2):
        """
        Initialize polynomial voltage profile.

        Args:
            order: Polynomial order (default: 2 for quadratic)
        """
        self.order = order
        self.n_parameters = (order + 1) * 2

        # Create mapping matrices
        self.theta_indices = list(range(0, self.n_parameters, 2))
        self.V_indices = list(range(1, self.n_parameters, 2))

    def evaluate(self, time, parameters):
        """
        Evaluate voltage at given time points.

        Args:
            time: Time values (shape: [n_samples, 1] or [n_samples])
            parameters: Voltage parameters (shape: [n_samples, n_parameters])

        Returns:
            theta: Voltage angle (radians)
            V: Voltage magnitude (p.u.)
        """
        is_torch = torch.is_tensor(time)

        # Ensure correct shapes
        if len(time.shape) == 1:
            time = time.reshape(-1, 1)
        if len(parameters.shape) == 1:
            parameters = parameters.reshape(1, -1)

        # Compute time powers [1, t, t^2, ..., t^order]
        if is_torch:
            time_powers = torch.cat([time ** i for i in range(self.order + 1)], dim=-1)
        else:
            time_powers = np.concatenate([time ** i for i in range(self.order + 1)], axis=-1)

        # Extract theta and V parameters
        theta_params = parameters[..., self.theta_indices]
        V_params = parameters[..., self.V_indices]

        # Evaluate polynomials
        if is_torch:
            theta = torch.sum(time_powers * theta_params, dim=-1, keepdim=True)
            V = torch.sum(time_powers * V_params, dim=-1, keepdim=True)
        else:
            theta = np.sum(time_powers * theta_params, axis=-1, keepdims=True)
            V = np.sum(time_powers * V_params, axis=-1, keepdims=True)

        return theta, V

    def get_initial_voltage(self, parameters):
        """
        Get voltage at t=0.

        Args:
            parameters: Voltage parameters

        Returns:
            theta_0: Initial voltage angle
            V_0: Initial voltage magnitude
        """
        theta_0 = parameters[..., 0:1]
        V_0 = parameters[..., 1:2]
        return theta_0, V_0

    def __repr__(self):
        return f"VoltageProfilePolynomial(order={self.order}, n_parameters={self.n_parameters})"
