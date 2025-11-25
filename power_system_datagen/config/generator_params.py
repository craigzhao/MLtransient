"""
Generator parameter configurations.

Contains predefined generator parameters for common test systems.
"""

import numpy as np


# IEEE 9-bus system generator parameters
IEEE9_GENERATORS = {
    "ieee9_1": {
        "name": "IEEE 9-bus Generator 1",
        "H_s": 23.64,  # Inertia constant (seconds)
        "D_pu": 2.0,  # Damping coefficient (p.u.)
        "X_d_pu": 0.146,  # d-axis synchronous reactance
        "X_q_pu": 0.0969,  # q-axis synchronous reactance
        "X_d_prime_pu": 0.0608,  # d-axis transient reactance
        "X_q_prime_pu": 0.0608,  # q-axis transient reactance (classical model)
        "T_d0_prime_s": 8.96,  # d-axis open circuit time constant
        "T_q0_prime_s": 0.31,  # q-axis open circuit time constant
        "R_s_pu": 0.0,  # Stator resistance
        "f_0_Hz": 60.0,  # Nominal frequency
        "norm_to_scale_delta": 1.0,
        "norm_to_scale_omega": 1.0,
        "set_point": [0.716, 0.27, 0.0, 1.04],  # [P, Q, theta, V]
        "control_adjustment": [0.5, 1.0],  # [P_M multiplier, E_fd multiplier]
    },
    "ieee9_2": {
        "name": "IEEE 9-bus Generator 2",
        "H_s": 6.4,
        "D_pu": 2.0,
        "X_d_pu": 0.8958,
        "X_q_pu": 0.8645,
        "X_d_prime_pu": 0.1198,
        "X_q_prime_pu": 0.1198,
        "T_d0_prime_s": 6.0,
        "T_q0_prime_s": 0.535,
        "R_s_pu": 0.0,
        "f_0_Hz": 60.0,
        "norm_to_scale_delta": 1.0,
        "norm_to_scale_omega": 1.0,
        "set_point": [1.63, 0.067, 0.1719, 1.025],
        "control_adjustment": [1.0, 1.0],
    },
    "ieee9_3": {
        "name": "IEEE 9-bus Generator 3",
        "H_s": 3.01,
        "D_pu": 2.0,
        "X_d_pu": 1.3125,
        "X_q_pu": 1.2578,
        "X_d_prime_pu": 0.1813,
        "X_q_prime_pu": 0.1813,
        "T_d0_prime_s": 5.89,
        "T_q0_prime_s": 0.6,
        "R_s_pu": 0.0,
        "f_0_Hz": 60.0,
        "norm_to_scale_delta": 1.0,
        "norm_to_scale_omega": 1.0,
        "set_point": [0.85, -0.109, 0.0785, 1.025],
        "control_adjustment": [1.0, 1.0],
    },
}


def get_generator_parameters(generator_id="ieee9_1"):
    """
    Get generator parameters by ID.

    Args:
        generator_id: Generator identifier (e.g., "ieee9_1")

    Returns:
        Dictionary of generator parameters
    """
    if generator_id in IEEE9_GENERATORS:
        return IEEE9_GENERATORS[generator_id].copy()
    else:
        raise ValueError(f"Unknown generator ID: {generator_id}. "
                         f"Available: {list(IEEE9_GENERATORS.keys())}")


def create_custom_generator(H, D, X_d, X_q, X_d_prime, X_q_prime,
                              T_d0_prime, T_q0_prime, R_s=0.0, f_0=60.0,
                              set_point=None, name="Custom Generator"):
    """
    Create custom generator parameters.

    Args:
        H: Inertia constant (seconds)
        D: Damping coefficient (p.u.)
        X_d: d-axis synchronous reactance (p.u.)
        X_q: q-axis synchronous reactance (p.u.)
        X_d_prime: d-axis transient reactance (p.u.)
        X_q_prime: q-axis transient reactance (p.u.)
        T_d0_prime: d-axis time constant (s)
        T_q0_prime: q-axis time constant (s)
        R_s: Stator resistance (p.u.)
        f_0: Nominal frequency (Hz)
        set_point: Operating point [P, Q, theta, V]
        name: Generator name

    Returns:
        Dictionary of generator parameters
    """
    return {
        "name": name,
        "H_s": H,
        "D_pu": D,
        "X_d_pu": X_d,
        "X_q_pu": X_q,
        "X_d_prime_pu": X_d_prime,
        "X_q_prime_pu": X_q_prime,
        "T_d0_prime_s": T_d0_prime,
        "T_q0_prime_s": T_q0_prime,
        "R_s_pu": R_s,
        "f_0_Hz": f_0,
        "norm_to_scale_delta": 1.0,
        "norm_to_scale_omega": 1.0,
        "set_point": set_point if set_point else [1.0, 0.0, 0.0, 1.0],
        "control_adjustment": [1.0, 1.0],
    }
