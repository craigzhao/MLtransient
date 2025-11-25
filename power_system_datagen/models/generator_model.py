"""
Synchronous Generator Model

Classical model with 4 states:
- E_q_prime: Transient EMF in q-axis
- E_d_prime: Transient EMF in d-axis
- delta: Rotor angle
- omega: Angular frequency deviation
"""

import torch
import numpy as np


class SynchronousGenerator:
    """
    Synchronous generator model based on classical machine dynamics.

    States: [E_q_prime, E_d_prime, delta, omega]
    Control inputs: [P_M, E_fd] (mechanical power, field voltage)

    Parameters:
        H: Inertia constant (seconds)
        D: Damping coefficient (p.u.)
        X_d, X_q: d-axis and q-axis synchronous reactances (p.u.)
        X_d_prime, X_q_prime: Transient reactances (p.u.)
        T_d0_prime, T_q0_prime: d-axis and q-axis open circuit time constants (s)
        R_s: Stator resistance (p.u.)
        f_0: Nominal frequency (Hz)
    """

    def __init__(self, parameters):
        """
        Initialize synchronous generator model.

        Args:
            parameters: Dictionary containing generator parameters
        """
        self.params = parameters

        # Extract parameters
        self.H = parameters["H_s"]
        self.D = parameters["D_pu"]
        self.X_d = parameters["X_d_pu"]
        self.X_q = parameters["X_q_pu"]
        self.X_d_prime = parameters["X_d_prime_pu"]
        self.X_q_prime = parameters["X_q_prime_pu"]
        self.T_d0_prime = parameters["T_d0_prime_s"]
        self.T_q0_prime = parameters["T_q0_prime_s"]
        self.R_s = parameters["R_s_pu"]
        self.f_0 = parameters.get("f_0_Hz", 60.0)

        self.omega_s = 2 * np.pi * self.f_0

        # State dimensions
        self.n_states = 4
        self.n_control_inputs = 2
        self.state_names = ["E_q_prime", "E_d_prime", "delta", "omega"]

        # Classical model assumption
        self.X_q = self.X_d_prime
        self.X_q_prime = self.X_d_prime

        # Impedance matrix for current calculation
        self.Z = np.array([[self.R_s, -self.X_q_prime],
                           [self.X_d_prime, self.R_s]])
        self.Z_inv = np.linalg.inv(self.Z)

        # Scaling parameters
        self.norm_to_scale_delta = parameters.get("norm_to_scale_delta", 1.0)
        self.norm_to_scale_omega = parameters.get("norm_to_scale_omega", 1.0)

    def dynamics(self, t, state, control_input, theta, V):
        """
        Compute state derivatives dx/dt.

        Args:
            t: Time (scalar or array)
            state: State vector [E_q_prime, E_d_prime, delta, omega]
            control_input: Control vector [P_M, E_fd]
            theta: Voltage angle at terminal (radians)
            V: Voltage magnitude at terminal (p.u.)

        Returns:
            State derivatives [dE_q/dt, dE_d/dt, ddelta/dt, domega/dt]
        """
        # Unpack states
        E_q_prime = state[..., 0:1]
        E_d_prime = state[..., 1:2]
        delta = state[..., 2:3]
        omega = state[..., 3:4]

        # Unpack controls
        P_M = control_input[..., 0:1]
        E_fd = control_input[..., 1:2]

        # Compute currents in d-q frame
        I_d, I_q = self._compute_current_dq(E_q_prime, E_d_prime, delta, theta, V)

        # Compute electrical power
        P_e = self._compute_electrical_power(E_q_prime, E_d_prime, I_d, I_q)

        # State derivatives
        dE_q_dt = torch.zeros_like(E_q_prime) if torch.is_tensor(E_q_prime) else np.zeros_like(E_q_prime)
        dE_d_dt = torch.zeros_like(E_d_prime) if torch.is_tensor(E_d_prime) else np.zeros_like(E_d_prime)
        ddelta_dt = omega * self.omega_s
        domega_dt = (P_M - P_e - self.D * omega) / (2 * self.H)

        if torch.is_tensor(state):
            return torch.cat([dE_q_dt, dE_d_dt, ddelta_dt, domega_dt], dim=-1)
        else:
            return np.concatenate([dE_q_dt, dE_d_dt, ddelta_dt, domega_dt], axis=-1)

    def _compute_current_dq(self, E_q_prime, E_d_prime, delta, theta, V):
        """Compute currents in d-q reference frame."""
        if torch.is_tensor(delta):
            v1 = E_d_prime - V * torch.sin(delta - theta)
            v2 = E_q_prime - V * torch.cos(delta - theta)
            I_d = self.Z_inv[0, 0] * v1 + self.Z_inv[0, 1] * v2
            I_q = self.Z_inv[1, 0] * v1 + self.Z_inv[1, 1] * v2
        else:
            v1 = E_d_prime - V * np.sin(delta - theta)
            v2 = E_q_prime - V * np.cos(delta - theta)
            I_d = self.Z_inv[0, 0] * v1 + self.Z_inv[0, 1] * v2
            I_q = self.Z_inv[1, 0] * v1 + self.Z_inv[1, 1] * v2
        return I_d, I_q

    def _compute_electrical_power(self, E_q_prime, E_d_prime, I_d, I_q):
        """Compute electrical power output."""
        P_e = (E_d_prime * I_d + E_q_prime * I_q +
               (self.X_q_prime - self.X_d_prime) * I_d * I_q)
        return P_e

    def compute_equilibrium(self, set_point):
        """
        Compute equilibrium state and control input from set point.

        Args:
            set_point: Array [P, Q, theta, V] - desired operating point

        Returns:
            equilibrium_state: Equilibrium state vector
            equilibrium_control: Equilibrium control input vector
        """
        is_torch = torch.is_tensor(set_point)

        if is_torch:
            P, Q, theta, V = set_point[..., 0:1], set_point[..., 1:2], set_point[..., 2:3], set_point[..., 3:4]
            V_complex = V * torch.exp(1j * theta)
            current_injection = (P - 1j * Q) / V_complex.conj()
            V_rotor = V_complex + (self.R_s + 1j * self.X_q) * current_injection
            delta = torch.angle(V_rotor)

            I_dq = current_injection * torch.exp(-1j * (delta - np.pi / 2))
            V_dq = V_complex * torch.exp(-1j * (delta - np.pi / 2))

            E_d_prime = (self.X_q - self.X_q_prime) * I_dq.imag
            E_q_prime = V_dq.imag + self.R_s * I_dq.imag + self.X_d_prime * I_dq.real
            E_fd = E_q_prime + (self.X_d - self.X_d_prime) * I_dq.real
            P_M = self._compute_electrical_power(E_q_prime, E_d_prime, I_dq.real, I_dq.imag)

            equilibrium_state = torch.cat([E_q_prime, E_d_prime, delta, torch.zeros_like(delta)], dim=-1)
            equilibrium_control = torch.cat([P_M, E_fd], dim=-1)
        else:
            P, Q, theta, V = set_point[..., 0:1], set_point[..., 1:2], set_point[..., 2:3], set_point[..., 3:4]
            V_complex = V * np.exp(1j * theta)
            current_injection = (P - 1j * Q) / V_complex.conj()
            V_rotor = V_complex + (self.R_s + 1j * self.X_q) * current_injection
            delta = np.angle(V_rotor)

            I_dq = current_injection * np.exp(-1j * (delta - np.pi / 2))
            V_dq = V_complex * np.exp(-1j * (delta - np.pi / 2))

            E_d_prime = (self.X_q - self.X_q_prime) * I_dq.imag
            E_q_prime = V_dq.imag + self.R_s * I_dq.imag + self.X_d_prime * I_dq.real
            E_fd = E_q_prime + (self.X_d - self.X_d_prime) * I_dq.real
            P_M = self._compute_electrical_power(E_q_prime, E_d_prime, I_dq.real, I_dq.imag)

            equilibrium_state = np.concatenate([E_q_prime, E_d_prime, delta, np.zeros_like(delta)], axis=-1)
            equilibrium_control = np.concatenate([P_M, E_fd], axis=-1)

        return equilibrium_state, equilibrium_control
