#!/usr/bin/env python3
"""
Produce PointCountFM training file from a raw ShowerData HDF5.

Uses showerdata.observables.calc_num_points_per_layer to compute per-layer
hit counts, then saves energy / num_points (N,45) / sampling_fraction /
n_layers in the format expected by MultiCaloDataLoader.

Usage:
    python src/preprocess_simplebox.py \
        --input  /path/to/4Mshowers_angles.h5 \
        --output data/SimpleBox_pretraining_4M.h5 \
        [--num-layers 45] [--batch-size 5000] [--max-showers N]
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from showerdata import core
from showerdata.observables import calc_num_points_per_layer
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", required=True, help="Source raw HDF5 (showers, energies, …)"
    )
    p.add_argument("--output", required=True, help="Output PointCountFM HDF5")
    p.add_argument(
        "--num-layers", type=int, default=45, help="Fixed output layer dim (default 45)"
    )
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument(
        "--max-showers", type=int, default=None, help="Limit showers (debug)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Input:      {src}")
    print(f"Output:     {dst}")
    print(f"Num layers: {args.num_layers}")

    with h5py.File(src, "r") as f:
        n_total = f["showers"].shape[0]
        n = min(n_total, args.max_showers) if args.max_showers else n_total
        print(f"Showers:    {n:,} / {n_total:,}")

        # Pre-allocate output arrays
        energy = np.empty((n, 1), dtype=np.float32)
        sampling_fraction = np.empty((n, 1), dtype=np.float32)
        n_layers_arr = np.empty((n, 1), dtype=np.int32)
        directions_arr = np.empty((n, 3), dtype=np.float32)
        num_points = np.zeros((n, args.num_layers), dtype=np.int32)

        for start in tqdm(range(0, n, args.batch_size), desc="Batches"):
            end = min(start + args.batch_size, n)
            sl = slice(start, end)

            points = core._get_shower_data(f, "showers", sl)
            energies = core._get_float_data(f, "energies", sl)
            pdg = core._get_int_data(f, "pdg", sl)
            directions = core._get_float_data(f, "directions", sl)
            shower_ids = core._get_int_data(f, "shower_ids", sl)

            showers = core.Showers(
                points=points,
                energies=energies,
                pdg=pdg,
                directions=directions,
                shower_ids=shower_ids,
            )

            counts = calc_num_points_per_layer(showers, num_layers=args.num_layers)
            num_points[start:end, : counts.shape[1]] = counts[:, : args.num_layers]

            energy[start:end, 0] = energies[:, 0]
            sampling_fraction[start:end, 0] = f["sampling_fraction"][sl][:, 0]
            n_layers_arr[start:end, 0] = f["num_layers"][sl][:, 0]
            directions_arr[start:end] = directions

    print(f"\nSaving → {dst} …", end=" ", flush=True)
    with h5py.File(dst, "w") as f:
        f.create_dataset("energy", data=energy, compression="gzip")
        f.create_dataset("num_points", data=num_points, compression="gzip")
        f.create_dataset(
            "sampling_fraction", data=sampling_fraction, compression="gzip"
        )
        f.create_dataset("n_layers", data=n_layers_arr, compression="gzip")
        f.create_dataset("directions", data=directions_arr, compression="gzip")
        f.attrs["n_showers"] = n
        f.attrs["num_layers"] = args.num_layers
        f.attrs["source"] = str(src)
    print("done")
    print(f"Total time: {time.time() - t0:.1f}s")
    print("\nOutput fields:")
    with h5py.File(dst, "r") as f:
        for k in f.keys():
            print(f"  {k}: {f[k].shape}")


if __name__ == "__main__":
    main()
