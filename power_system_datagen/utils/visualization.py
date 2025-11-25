"""
Visualization utilities for datasets.

Functions for plotting trajectories and analyzing datasets.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_trajectories(dataset, n_trajectories=5, state_names=None, figsize=(12, 8)):
    """
    Plot sample trajectories from dataset.

    Args:
        dataset: Dataset dictionary
        n_trajectories: Number of trajectories to plot
        state_names: List of state names
        figsize: Figure size
    """
    # Extract data
    time = dataset["time"]
    states_initial = dataset["state_initial"]
    states_final = dataset.get("state_result", None)

    if torch.is_tensor(time):
        time = time.numpy()
    if torch.is_tensor(states_initial):
        states_initial = states_initial.numpy()
    if states_final is not None and torch.is_tensor(states_final):
        states_final = states_final.numpy()

    n_states = states_initial.shape[1]
    if state_names is None:
        state_names = [f"State {i}" for i in range(n_states)]

    # Create subplots
    fig, axes = plt.subplots(n_states, 1, figsize=figsize)
    if n_states == 1:
        axes = [axes]

    # Select random trajectories
    n_samples = time.shape[0]
    indices = np.random.choice(n_samples, min(n_trajectories, n_samples), replace=False)

    for i, ax in enumerate(axes):
        for idx in indices:
            t = time[idx, 0]
            x0 = states_initial[idx, i]

            if states_final is not None:
                xf = states_final[idx, i]
                ax.plot([0, t], [x0, xf], 'o-', alpha=0.6, markersize=4)
            else:
                ax.plot([0, t], [x0, x0], 'o', alpha=0.6, markersize=4)

        ax.set_ylabel(state_names[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Sample Trajectories")
    plt.tight_layout()
    return fig


def plot_dataset_statistics(dataset, state_names=None, figsize=(14, 10)):
    """
    Plot statistical distributions of dataset.

    Args:
        dataset: Dataset dictionary
        state_names: List of state names
        figsize: Figure size
    """
    # Extract data
    time = dataset["time"]
    states_initial = dataset["state_initial"]
    states_final = dataset.get("state_result", None)
    voltage_params = dataset.get("voltage_parametrisation", None)

    if torch.is_tensor(time):
        time = time.numpy()
    if torch.is_tensor(states_initial):
        states_initial = states_initial.numpy()
    if states_final is not None and torch.is_tensor(states_final):
        states_final = states_final.numpy()
    if voltage_params is not None and torch.is_tensor(voltage_params):
        voltage_params = voltage_params.numpy()

    n_states = states_initial.shape[1]
    if state_names is None:
        state_names = [f"State {i}" for i in range(n_states)]

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Time distribution
    ax = plt.subplot(3, 3, 1)
    ax.hist(time.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Time Distribution")
    ax.grid(True, alpha=0.3)

    # Initial state distributions
    for i in range(min(4, n_states)):
        ax = plt.subplot(3, 3, 2 + i)
        ax.hist(states_initial[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel(state_names[i])
        ax.set_ylabel("Count")
        ax.set_title(f"Initial {state_names[i]}")
        ax.grid(True, alpha=0.3)

    # State space plot (delta vs omega)
    if n_states >= 4:
        ax = plt.subplot(3, 3, 7)
        ax.scatter(states_initial[:, 2], states_initial[:, 3],
                   alpha=0.5, s=10, c=time.flatten(), cmap='viridis')
        ax.set_xlabel("delta (rad)")
        ax.set_ylabel("omega (p.u.)")
        ax.set_title("Initial State Space")
        ax.grid(True, alpha=0.3)
        plt.colorbar(ax.collections[0], ax=ax, label="Time (s)")

    # Voltage parameter distributions
    if voltage_params is not None:
        for i in range(min(2, voltage_params.shape[1])):
            ax = plt.subplot(3, 3, 8 + i)
            ax.hist(voltage_params[:, i], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel(f"V_param_{i}")
            ax.set_ylabel("Count")
            ax.set_title(f"Voltage Parameter {i}")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_phase_portrait(dataset, state_idx1=2, state_idx2=3,
                         state_names=None, figsize=(8, 6)):
    """
    Plot phase portrait of two states.

    Args:
        dataset: Dataset dictionary
        state_idx1: First state index
        state_idx2: Second state index
        state_names: List of state names
        figsize: Figure size
    """
    states_initial = dataset["state_initial"]
    states_final = dataset.get("state_result", None)

    if torch.is_tensor(states_initial):
        states_initial = states_initial.numpy()
    if states_final is not None and torch.is_tensor(states_final):
        states_final = states_final.numpy()

    if state_names is None:
        state_names = [f"State {i}" for i in range(states_initial.shape[1])]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot initial points
    ax.scatter(states_initial[:, state_idx1], states_initial[:, state_idx2],
               alpha=0.5, s=20, c='blue', label='Initial')

    # Plot final points if available
    if states_final is not None:
        ax.scatter(states_final[:, state_idx1], states_final[:, state_idx2],
                   alpha=0.5, s=20, c='red', label='Final')

        # Plot trajectories
        for i in range(0, len(states_initial), max(1, len(states_initial) // 20)):
            ax.plot([states_initial[i, state_idx1], states_final[i, state_idx1]],
                    [states_initial[i, state_idx2], states_final[i, state_idx2]],
                    'k-', alpha=0.2, linewidth=0.5)

    ax.set_xlabel(state_names[state_idx1])
    ax.set_ylabel(state_names[state_idx2])
    ax.set_title("Phase Portrait")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
