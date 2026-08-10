"""
Preprocess lemurs_4M.h5 into a PointCountFM-compatible training file.

Input:  AllShowers-AllGeometries/data/pretrain_lemurs/lemurs_4M.h5
        Combined LEMURS dataset, 4 detectors x 1M showers (4M total),
        stored as 1M-shower contiguous blocks in attrs['detectors'] order:
            block 0  par04_siw   layers=90
            block 1  par04_scipb layers=45
            block 2  odd         layers=48
            block 3  fccee_cld   layers=40
        Keys: showers (object, ragged flat (x,y,z,E,t) per hit, stride=5),
              energies (N,1), num_layers (N,1), sampling_fraction (N,1),
              directions (N,3), layer_z_pos (N,90), num_points (N,) total,
              pdg, shower_ids, gun_position
        Layer index is stored at shower_flat[2::5] as an integer (0-based).

Output: PointCountFM/data/LEMURS_pretraining_4M.h5
        Schema matches preprocess_allegro.py / SimpleBox_pretraining_4M.h5:
            energy           (N, 1)  float32
            n_layers         (N, 1)  int32
            num_points       (N, 90) int32   <-- per-layer counts, padded to 90
            sampling_fraction(N, 1)  float32
            directions       (N, 3)  float32
        No detector-id field (PointCountFM dataset.py expects exactly the
        five keys above; conditioning is on energy + sf + n_layers + dir).
"""

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm

SOURCE = "data/lemurs_4M.h5"
OUTPUT = "data/LEMURS_pretraining_4M.h5"

MAX_LAYERS = 90
VALS_PER_HIT = 5  # (x, y, z=layer_idx, E, t) stored flat per shower


def compute_per_layer_counts(
    showers: h5py.Dataset,
    num_points: np.ndarray,
) -> np.ndarray:
    """
    For each shower, count hits per layer.

    The third column of the flat hit array (index 2, stride 5) stores the
    layer index as an integer (0-based) — NOT a physical z-coordinate.
    np.bincount with minlength=MAX_LAYERS produces the per-layer histogram
    padded to MAX_LAYERS for detectors with fewer active layers.

    Returns array of shape (N, MAX_LAYERS) with integer hit counts.
    """
    N = len(showers)
    result = np.zeros((N, MAX_LAYERS), dtype=np.int32)

    for i in tqdm(range(N), desc="Extracting per-layer hit counts"):
        n_pts = int(num_points[i])
        if n_pts == 0:
            continue
        shower_flat = showers[i]
        layer_indices = shower_flat[2::VALS_PER_HIT].astype(int)
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
        attrs = dict(src.attrs)
        print(f"\nShowers : {N:,}")
        print(f"Attrs   : {attrs}")

        energies = src["energies"][:]  # (N, 1)
        num_layers = src["num_layers"][:]  # (N, 1)
        sampling_fraction = src["sampling_fraction"][:]  # (N, 1)
        directions = src["directions"][:]  # (N, 3)
        num_points_raw = src["num_points"][:]  # (N,) — total hits per shower

        # Per-detector breakdown (using contiguous 1M blocks, attrs['detectors'] order)
        det_names = [
            d.decode() if isinstance(d, bytes) else str(d)
            for d in attrs.get("detectors", [])
        ]
        block_size = N // len(det_names) if det_names else N
        print(f"\nPer-detector breakdown (block size = {block_size:,}):")
        for i, name in enumerate(det_names):
            s, e = i * block_size, (i + 1) * block_size
            nl_unique = np.unique(num_layers[s:e].flatten())
            sf_unique = np.unique(sampling_fraction[s:e].flatten())
            pts_block = num_points_raw[s:e]
            print(
                f"  [{i}] {name:<12} n={e - s:,} | "
                f"n_layers={nl_unique} | sf={sf_unique} | "
                f"pts/shower min={pts_block.min()} max={pts_block.max()}"
            )

        print(
            f"\nGlobal energy range  : [{energies.min():.2f}, {energies.max():.2f}] GeV"
        )
        print(
            f"Global hits/shower   : [{num_points_raw.min()}, {num_points_raw.max()}]"
        )

        num_points_per_layer = compute_per_layer_counts(src["showers"], num_points_raw)

    total_hits_computed = num_points_per_layer.sum(axis=1)
    mismatches = np.sum(total_hits_computed != num_points_raw)
    if mismatches > 0:
        print(
            f"WARNING: {mismatches} showers have total-hit mismatch "
            f"(layer index outside [0, {MAX_LAYERS - 1}]?)."
        )
    else:
        print("Hit count check: OK (all totals match)")

    print(
        f"\nGlobal max points-per-layer (across all showers, all layers) : "
        f"{num_points_per_layer.max()}"
    )

    # Per-detector post-binning summary
    if det_names:
        print("\nPer-detector post-binning summary:")
        for i, name in enumerate(det_names):
            s, e = i * block_size, (i + 1) * block_size
            block = num_points_per_layer[s:e]
            active_layers = (block.sum(axis=0) > 0).sum()
            print(
                f"  [{i}] {name:<12} active_layer_dims={active_layers}/{MAX_LAYERS} "
                f"max_pts_per_layer={block.max()}"
            )

    os.makedirs(os.path.dirname(output), exist_ok=True)
    print(f"\nSaving to {output} ...")
    with h5py.File(output, "w") as dst:
        dst.create_dataset("energy", data=energies, compression="gzip")
        dst.create_dataset(
            "n_layers", data=num_layers.astype(np.int32), compression="gzip"
        )
        dst.create_dataset("num_points", data=num_points_per_layer, compression="gzip")
        dst.create_dataset(
            "sampling_fraction", data=sampling_fraction, compression="gzip"
        )
        dst.create_dataset("directions", data=directions, compression="gzip")
        # Preserve provenance — not consumed by dataset.py, ignored on load.
        dst.attrs["source"] = source
        dst.attrs["max_layers"] = MAX_LAYERS
        if det_names:
            dst.attrs["detectors"] = np.array(det_names, dtype=h5py.string_dtype())
            dst.attrs["block_size"] = block_size

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess LEMURS combined dataset for PointCountFM."
    )
    parser.add_argument("--source", default=SOURCE, help="Input HDF5 file.")
    parser.add_argument("--output", default=OUTPUT, help="Output HDF5 file.")
    args = parser.parse_args()
    main(args.source, args.output)
