"""
Preprocess allegro_100k.h5 into a PointCountFM-compatible training file.

Input:  AllShowers-AllGeometries/data/allegro_100k.h5
        Keys: showers (object, ragged), energies (N,1), num_layers (N,1),
              sampling_fraction (N,1), directions (N,3), layer_z_pos (N,45)

Output: AllShowers-AllGeometries/data/allegro_100k_pcfm.h5
        Keys: energy (N,1), n_layers (N,1), num_points (N,45),
              sampling_fraction (N,1), directions (N,3)

Per-layer hit counts are extracted from raw shower hits (x,y,z,E,t per hit)
by assigning each hit to its nearest layer via the layer z-positions.
"""

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm

SOURCE = "data/allegro_100k.h5"
OUTPUT = "data/allegro_100k_pcfm.h5"

MAX_LAYERS = 45
VALS_PER_HIT = 5  # (x, y, z, E, t) stored flat per shower


def compute_per_layer_counts(
    showers: h5py.Dataset,
    num_points: np.ndarray,
    layer_z_pos: np.ndarray,
) -> np.ndarray:
    """
    For each shower, count hits per layer.

    The third column of the flat hit array (index 2, stride 5) stores the
    layer index as an integer (0, 1, 2, ...) — NOT a physical z-coordinate.
    We use it directly to accumulate per-layer hit counts.

    Returns array of shape (N, MAX_LAYERS) with integer hit counts.
    """
    N = len(showers)
    result = np.zeros((N, MAX_LAYERS), dtype=np.int32)

    for i in tqdm(range(N), desc="Extracting per-layer hit counts"):
        n_pts = int(num_points[i])
        if n_pts == 0:
            continue
        shower_flat = showers[i]  # shape (n_pts * VALS_PER_HIT,)
        layer_indices = shower_flat[2::VALS_PER_HIT].astype(
            int
        )  # integer layer index per hit

        counts = np.bincount(layer_indices, minlength=MAX_LAYERS)
        result[i] = counts[:MAX_LAYERS]

    return result


def main(source: str = SOURCE, output: str = OUTPUT) -> None:
    print(f"Source : {source}")
    print(f"Output : {output}")

    if os.path.exists(output):
        print("Output already exists — delete it to reprocess.")
        return

    with h5py.File(source, "r") as src:
        N = len(src["energies"])
        print(f"\nShowers : {N:,}")

        energies = src["energies"][:]  # (N, 1)
        num_layers = src["num_layers"][:]  # (N, 1)  — all 11 for Allegro
        sampling_fraction = src["sampling_fraction"][:]  # (N, 1)
        directions = src["directions"][:]  # (N, 3)
        num_points_raw = src["num_points"][:]  # (N,) — total hits per shower
        layer_z_pos = src["layer_z_pos"][:]  # (N, 45) — z-positions of layers

        print(f"Energy range  : [{energies.min():.2f}, {energies.max():.2f}] GeV")
        print(f"n_layers      : {np.unique(num_layers.flatten())}")
        print(f"sampling_frac : {np.unique(sampling_fraction.flatten())}")
        print(f"hits/shower   : [{num_points_raw.min()}, {num_points_raw.max()}]")

        num_points_per_layer = compute_per_layer_counts(
            src["showers"], num_points_raw, layer_z_pos
        )

    total_hits_computed = num_points_per_layer.sum(axis=1)
    mismatches = np.sum(total_hits_computed != num_points_raw)
    if mismatches > 0:
        print(
            f"WARNING: {mismatches} showers have total-hit mismatch (check z-assignment)."
        )
    else:
        print("Hit count check: OK (all totals match)")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"\nSaving to {output} ...")
    with h5py.File(output, "w") as dst:
        dst.create_dataset("energy", data=energies, compression="gzip")
        dst.create_dataset("n_layers", data=num_layers, compression="gzip")
        dst.create_dataset("num_points", data=num_points_per_layer, compression="gzip")
        dst.create_dataset(
            "sampling_fraction", data=sampling_fraction, compression="gzip"
        )
        dst.create_dataset("directions", data=directions, compression="gzip")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Allegro data for PointCountFM."
    )
    parser.add_argument("--source", default=SOURCE, help="Input HDF5 file.")
    parser.add_argument("--output", default=OUTPUT, help="Output HDF5 file.")
    args = parser.parse_args()
    main(args.source, args.output)
