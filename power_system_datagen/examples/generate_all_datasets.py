"""
Generate all standard datasets for IEEE 9-bus system.

Creates train/test/collocation datasets for all three generators.
"""

import sys
sys.path.insert(0, '..')

from power_system_datagen import DataGenerator
from power_system_datagen.config import DatasetConfig


def main():
    print("=" * 70)
    print("Generating All Standard Datasets for IEEE 9-Bus System")
    print("=" * 70)

    generators = ["ieee9_1", "ieee9_2", "ieee9_3"]
    dataset_types = ["train", "test", "collocation"]

    seeds = {
        "ieee9_1": {"train": 192444352, "test": 28374651, "collocation": 87654321},
        "ieee9_2": {"train": 73829461, "test": 19283746, "collocation": 23456789},
        "ieee9_3": {"train": 56473829, "test": 98765432, "collocation": 34567890},
    }

    for gen_id in generators:
        print(f"\n{'=' * 70}")
        print(f"Generator: {gen_id}")
        print(f"{'=' * 70}")

        for dtype in dataset_types:
            print(f"\n[{dtype.upper()}] Generating dataset...")

            # Create config
            if dtype == "train":
                config = DatasetConfig.training_config(
                    n_samples=2500,
                    random_seed=seeds[gen_id][dtype]
                )
            elif dtype == "test":
                config = DatasetConfig.test_config(
                    n_samples=4000,
                    random_seed=seeds[gen_id][dtype]
                )
            else:  # collocation
                config = DatasetConfig.collocation_config(
                    n_samples=10000,
                    random_seed=seeds[gen_id][dtype]
                )

            # Generate dataset
            generator = DataGenerator(gen_id, config)
            dataset = generator.generate_dataset(verbose=True)

            # Save
            filename = f"../data/{dtype}_{gen_id}.pkl"
            generator.save(dataset, filename)

    print("\n" + "=" * 70)
    print("All datasets generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
