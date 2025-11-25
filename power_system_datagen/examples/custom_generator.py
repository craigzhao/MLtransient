"""
Example using custom generator parameters.

Demonstrates how to define your own generator parameters.
"""

import sys
sys.path.insert(0, '..')

from power_system_datagen import DataGenerator
from power_system_datagen.config import DatasetConfig, create_custom_generator


def main():
    print("=" * 70)
    print("Custom Generator Example")
    print("=" * 70)

    # =========================================================================
    # Define custom generator parameters
    # =========================================================================
    print("\n[1] Creating custom generator parameters...")

    custom_params = create_custom_generator(
        H=5.0,              # Inertia constant
        D=2.0,              # Damping
        X_d=1.0,            # d-axis reactance
        X_q=0.6,            # q-axis reactance
        X_d_prime=0.3,      # d-axis transient reactance
        X_q_prime=0.3,      # q-axis transient reactance
        T_d0_prime=6.0,     # d-axis time constant
        T_q0_prime=0.5,     # q-axis time constant
        R_s=0.0,            # Stator resistance
        f_0=60.0,           # Nominal frequency
        set_point=[1.0, 0.2, 0.0, 1.0],  # [P, Q, theta, V]
        name="My Custom Generator"
    )

    print(f"    Created: {custom_params['name']}")
    print(f"    Set point: P={custom_params['set_point'][0]}, "
          f"Q={custom_params['set_point'][1]}, "
          f"V={custom_params['set_point'][3]}")

    # =========================================================================
    # Create dataset configuration
    # =========================================================================
    print("\n[2] Creating dataset configuration...")

    config = DatasetConfig(
        n_samples=1000,
        t_min=0.0,
        t_max=1.0,
        time_sampling="linear",
        perturbation_radius=0.3,
        state_sampling="annular",
        simulate=True,
        random_seed=123
    )

    # =========================================================================
    # Generate dataset
    # =========================================================================
    print("\n[3] Generating dataset with custom generator...")

    generator = DataGenerator(custom_params, config)
    dataset = generator.generate_dataset(verbose=True)

    # =========================================================================
    # Save dataset
    # =========================================================================
    print("\n[4] Saving dataset...")
    generator.save(dataset, "../data/custom_generator_dataset.pkl")

    print("\n" + "=" * 70)
    print("Custom generator example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
