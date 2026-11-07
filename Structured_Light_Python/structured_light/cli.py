"""Command-line entry point for the simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .simulation import SimulationConfig, run_simulation, save_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate and train a multi-layer structured-light D2NN."
    )
    parser.add_argument("--grid-size", type=int, default=600, help="Square simulation grid.")
    parser.add_argument("--modes", type=int, default=3, help="Number of input LG modes.")
    parser.add_argument("--screens", type=int, default=5, help="Number of phase screens.")
    parser.add_argument(
        "--phase-screen-size",
        type=int,
        help="Square phase-screen width in pixels (default: min(300, half the grid)).",
    )
    parser.add_argument(
        "--fourier-basis-size",
        type=int,
        default=11,
        help="Odd width of the complex Fourier coefficient grid (MATLAB default: 11).",
    )
    parser.add_argument(
        "--num-subharmonics",
        type=int,
        default=0,
        help="Number of fixed low-frequency subharmonic levels.",
    )
    parser.add_argument("--epochs", type=int, default=5000, help="SPSA training iterations.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e3,
        help="Initial SPSA step size (MATLAB Fourier default: 5000).",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.1,
        help="Initial SPSA perturbation (MATLAB Fourier default: 0.1).",
    )
    parser.add_argument(
        "--learning-rate-exponent",
        type=float,
        default=0.18,
        help="Power-law decay exponent for the SPSA step size.",
    )
    parser.add_argument(
        "--perturbation-exponent",
        type=float,
        default=0.101,
        help="Power-law decay exponent for the SPSA perturbation.",
    )
    parser.add_argument(
        "--failsafe-check",
        type=int,
        default=10,
        help="Evaluate optimizer recovery every N epochs.",
    )
    parser.add_argument(
        "--failsafe-tolerance",
        type=float,
        default=2e-4,
        help="Minimum cost improvement expected at a failsafe check.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for CSV, JSON, NPZ, and PNG output.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-epoch output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SimulationConfig(
        grid_size=args.grid_size,
        num_modes=args.modes,
        num_screens=args.screens,
        phase_screen_size=args.phase_screen_size,
        fourier_basis_size=args.fourier_basis_size,
        num_subharmonics=args.num_subharmonics,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        perturbation=args.perturbation,
        learning_rate_exponent=args.learning_rate_exponent,
        perturbation_exponent=args.perturbation_exponent,
        failsafe_check=args.failsafe_check,
        failsafe_tolerance=args.failsafe_tolerance,
        seed=args.seed,
    )
    result = run_simulation(config, verbose=not args.quiet)
    save_result(result, args.output_dir)
    print(f"Results saved to {args.output_dir.resolve()}")
    print("Final metrics:")
    for name, value in result.final_metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
