"""Command-line entry point for the simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .simulation import SimulationConfig, run_simulation, save_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate and train a multi-layer structured-light D2NN."
    )
    parser.add_argument("--grid-size", type=int, default=96, help="Square simulation grid.")
    parser.add_argument("--modes", type=int, default=3, help="Number of input LG modes.")
    parser.add_argument("--screens", type=int, default=3, help="Number of phase screens.")
    parser.add_argument(
        "--phase-screen-size",
        type=int,
        help="Square phase-screen width in pixels (default: half the simulation grid).",
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
    parser.add_argument("--epochs", type=int, default=10, help="SPSA training iterations.")
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
