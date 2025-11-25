"""
Basic usage example for power system data generation.

This script demonstrates how to:
1. Create a simple dataset
2. Visualize the results
3. Save and load data
"""

import sys
sys.path.insert(0, '..')

from power_system_datagen import DataGenerator
from power_system_datagen.config import DatasetConfig
from power_system_datagen.utils import load_dataset, print_dataset_summary
from power_system_datagen.utils.visualization import (
    plot_trajectories, plot_dataset_statistics, plot_phase_portrait
)
import matplotlib.pyplot as plt


def main():
    print("=" * 70)
    print("Power System Data Generation - Basic Example")
    print("=" * 70)

    # =========================================================================
    # Step 1: Create dataset configuration
    # =========================================================================
    print("\n[1] Creating dataset configuration...")

    config = DatasetConfig(
        n_samples=500,
        t_min=0.0,
        t_max=0.5,
        time_sampling="linear",
        perturbation_radius=0.25,
        state_sampling="annular",
        simulate=True,
        random_seed=42,
        use_torch=True,
        backend="scipy"
    )

    print(f"    Configuration: {config}")

    # =========================================================================
    # Step 2: Initialize generator and create dataset
    # =========================================================================
    print("\n[2] Initializing data generator...")

    # Use IEEE 9-bus generator 1 parameters
    generator = DataGenerator(
        generator_params="ieee9_1",
        dataset_config=config
    )

    print(f"    Generator: {generator.generator_params['name']}")
    print(f"    Equilibrium state: {generator.equilibrium_state}")
    print(f"    Control input: {generator.control_input}")

    # Generate dataset
    print("\n[3] Generating dataset...")
    dataset = generator.generate_dataset(verbose=True)

    # =========================================================================
    # Step 3: Inspect dataset
    # =========================================================================
    print("\n[4] Dataset summary:")
    print_dataset_summary(dataset)

    # =========================================================================
    # Step 4: Save dataset
    # =========================================================================
    print("\n[5] Saving dataset...")
    output_file = "../data/example_dataset.pkl"
    generator.save(dataset, output_file, format="pickle")

    # =========================================================================
    # Step 5: Load and verify
    # =========================================================================
    print("\n[6] Loading dataset to verify...")
    loaded_dataset = load_dataset(output_file, format="pickle", use_torch=True)
    print("    Dataset loaded successfully!")

    # =========================================================================
    # Step 6: Visualize results
    # =========================================================================
    print("\n[7] Creating visualizations...")

    # Plot trajectories
    fig1 = plot_trajectories(
        dataset,
        n_trajectories=10,
        state_names=["E_q'", "E_d'", "delta", "omega"]
    )
    fig1.savefig("../data/trajectories.png", dpi=150, bbox_inches='tight')
    print("    Saved: trajectories.png")

    # Plot dataset statistics
    fig2 = plot_dataset_statistics(
        dataset,
        state_names=["E_q'", "E_d'", "delta", "omega"]
    )
    fig2.savefig("../data/statistics.png", dpi=150, bbox_inches='tight')
    print("    Saved: statistics.png")

    # Plot phase portrait
    fig3 = plot_phase_portrait(
        dataset,
        state_idx1=2,  # delta
        state_idx2=3,  # omega
        state_names=["E_q'", "E_d'", "delta", "omega"]
    )
    fig3.savefig("../data/phase_portrait.png", dpi=150, bbox_inches='tight')
    print("    Saved: phase_portrait.png")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    main()
