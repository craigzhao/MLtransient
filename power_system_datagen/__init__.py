"""
Power System Data Generation Framework
========================================

A standalone framework for generating physics-based training data
for power system transient analysis using synchronous generator models.

Main components:
- Generator models (synchronous machine dynamics)
- Voltage profile parametrization
- Sampling strategies (time, states, voltage)
- ODE simulation
- Data generation pipeline
"""

__version__ = "1.0.0"

from .generator import DataGenerator
from .models.generator_model import SynchronousGenerator
from .models.voltage_profile import VoltageProfilePolynomial
from .config.generator_params import get_generator_parameters

__all__ = [
    "DataGenerator",
    "SynchronousGenerator",
    "VoltageProfilePolynomial",
    "get_generator_parameters",
]
