"""Structured-light simulation package."""

from .simulation import (
    SimulationConfig,
    SimulationResult,
    evaluate_coefficients,
    run_simulation,
)

__all__ = [
    "SimulationConfig",
    "SimulationResult",
    "evaluate_coefficients",
    "run_simulation",
]
