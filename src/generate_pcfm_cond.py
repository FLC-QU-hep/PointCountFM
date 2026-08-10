#!/usr/bin/env python3
"""generate_pcfm_cond.py

Replace `num_points_per_layer` in a conditioning HDF5 file with
per-event predictions from a trained PointCountFM model (+ bias correction).

Only events matching the target nl value are replaced; other events keep
their original Geant4 counts unchanged.

Usage:
    source AllShowers-AllGeometries/.venv/bin/activate
    cd PointCountFM

    python src/generate_pcfm_cond.py --downstream ALLEGRO
    python src/generate_pcfm_cond.py --downstream SimpleBox --scales 90k
    python src/generate_pcfm_cond.py --downstream SimpleBox --scales 90k --strategies finetune
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from pcfm_conditioning import build_conditions, load_pcfm_model
from pcfm_utils import (
    apply_bias_correction,
    heun_sampler,
    inverse_transform,
)
from preprocessing import detect_active_dims

REPO = Path(__file__).resolve().parent.parent
ALLSHOWERS = REPO.parent / "AllShowers-AllGeometries"

DOWNSTREAMS = {
    "ALLEGRO": {
        "cond_file": ALLSHOWERS / "data/finetune_allegro/allegro_10k_cond.h5",
        "pcfm_root": "results/finetune_ALLEGRO",
        "out_dir": ALLSHOWERS / "data/pcfm_cond/allegro",
        "nl_val": 11,
        "sf_val": 0.162,
        "n_active": 11,
        "scales": ["100", "1k", "10k", "100k"],
        "subdirs": {
            "finetune": "finetune_reset",
            "finetune_fallback": "finetune",
            "scratch": "from_scratch",
        },
    },
    "SimpleBox": {
        "cond_file": ALLSHOWERS / "data/finetune_simplebox/simplebox_10k_cond.h5",
        "pcfm_root": "results/finetune_SimpleBox",
        "out_dir": ALLSHOWERS / "data/pcfm_cond/simplebox",
        "nl_val": 35,
        "sf_val": 0.035,
        "n_active": 35,
        "scales": ["100", "1k", "10k", "90k"],
        "subdirs": {
            "finetune": "finetune",
            "scratch": "from_scratch",
        },
    },
    "CLD": {
        "cond_file": ALLSHOWERS / "data/finetune_cld/cld_10k_cond.h5",
        "pcfm_root": "results/finetune_CLD",
        "out_dir": ALLSHOWERS / "data/pcfm_cond/cld",
        "nl_val": 40,
        "sf_val": 0.0263,
        "n_active": 40,
        "scales": ["100", "1k", "10k", "100k"],
        "subdirs": {
            "finetune": "finetune",
            "scratch": "from_scratch",
        },
    },
    "ODD": {
        "cond_file": ALLSHOWERS / "data/finetune_odd/odd_10k_cond.h5",
        "pcfm_root": "results/finetune_ODD",
        "out_dir": ALLSHOWERS / "data/pcfm_cond/odd",
        "nl_val": 48,
        "sf_val": 0.0255,
        "n_active": 48,
        "scales": ["100", "1k", "10k", "100k"],
        "subdirs": {
            "finetune": "finetune",
            "scratch": "from_scratch",
        },
    },
    "Par04SciPb": {
        "cond_file": ALLSHOWERS / "data/finetune_par04_scipb/par04_scipb_10k_cond.h5",
        "pcfm_root": "results/finetune_Par04SciPb",
        "out_dir": ALLSHOWERS / "data/pcfm_cond/par04_scipb",
        "nl_val": 45,
        "sf_val": 0.033,
        "n_active": 45,
        "scales": ["1k", "10k", "100k"],
        "subdirs": {
            "finetune": "finetune",
            "scratch": "from_scratch",
        },
    },
    "Par04SiW": {
        "cond_file": ALLSHOWERS / "data/finetune_par04_siw/par04_siw_10k_cond.h5",
        "pcfm_root": "results/finetune_Par04SiW",
        "out_dir": ALLSHOWERS / "data/pcfm_cond/par04_siw",
        "nl_val": 90,
        "sf_val": 0.026,
        "n_active": 90,
        "scales": ["1k", "10k", "100k"],
        "subdirs": {
            "finetune": "finetune",
            "scratch": "from_scratch",
        },
    },
}


def generate_num_points(
    model,
    transform,
    norm_stats,
    config,
    energies_gev,
    dirs,
    nl_val,
    sf_val,
    device,
    n_steps=100,
    batch_size=1024,
    seed=42,
    temperature=1.0,
):
    N = len(energies_gev)
    dim_input = config["model"]["dim_input"]
    active_mask = detect_active_dims(transform, device)

    all_gen = []
    for bi, start in enumerate(range(0, N, batch_size)):
        end = min(start + batch_size, N)
        e_batch = energies_gev[start:end].flatten()
        d_batch = dirs[start:end] if dirs is not None else None

        cond = build_conditions(e_batch, d_batch, norm_stats, nl_val, sf_val, device)
        torch.manual_seed(seed + bi)
        x0 = torch.randn(end - start, dim_input, device=device)
        if active_mask is not None:
            x0 = x0 * active_mask

        with torch.no_grad():
            x = heun_sampler(model, x0, cond, n_steps, temperature=temperature)
        all_gen.append(inverse_transform(x, transform))

    return np.concatenate(all_gen, axis=0)


def refresh_bias_correction(model_dir: Path, mode: str = "1d", extra_args: list = None):
    """Recompute bias-correction npz if missing or stale relative to ckpt."""
    ckpt = model_dir / "best_physics_model.pt"
    if not ckpt.exists():
        ckpt = model_dir / "best_model.pt"
    if not ckpt.exists():
        return
    npz_name = "bias_correction_2d.npz" if mode == "2d" else "bias_correction.npz"
    npz = model_dir / npz_name
    if npz.exists() and npz.stat().st_mtime >= ckpt.stat().st_mtime:
        return
    print(f"    Recomputing bias correction (mode={mode}) for {model_dir.name} ...")
    script = Path(__file__).parent / "compute_bias_correction.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--result_dir",
            str(model_dir),
            "--bias-mode",
            mode,
            *(extra_args or []),
        ],
        check=True,
    )


def apply_bias_correction_2d(gen_counts, bc_path, energies_gev):
    """Per-(layer,energy-bin) BC. Bin each event by log10(E_GeV).

    F1: validates that the npz was produced by the 2D fit path
    (rejects 1D npz or other malformed schemas with a clear message).
    R3: warns to stderr if any event's log10(E) falls outside the fitted
    edges before clamping.
    """
    bc = np.load(bc_path, allow_pickle=False)
    if "bias_mode" not in bc.files:
        raise ValueError(
            f"{bc_path}: missing 'bias_mode' key — not a 2D bias-correction npz. "
            "Re-run compute_bias_correction.py with --bias-mode 2d."
        )
    mode_arr = bc["bias_mode"]
    mode = (
        mode_arr.item().decode()
        if isinstance(mode_arr.item(), bytes)
        else str(mode_arr.item())
    )
    if mode != "2d":
        raise ValueError(
            f"{bc_path}: bias_mode={mode!r}, expected '2d'. "
            "Re-run compute_bias_correction.py with --bias-mode 2d."
        )
    if "bias_factor_2d" not in bc.files or "e_bin_edges" not in bc.files:
        raise ValueError(
            f"{bc_path}: required keys missing (bias_factor_2d, e_bin_edges)."
        )
    bf = bc["bias_factor_2d"]  # [nl, N_BINS]
    edges = bc["e_bin_edges"]
    nl = bf.shape[0]
    log_e = np.log10(np.asarray(energies_gev, dtype=np.float64).flatten())
    n_below = int((log_e < edges[0]).sum())
    n_above = int((log_e > edges[-1]).sum())
    if n_below or n_above:
        sys.stderr.write(
            f"[apply_bias_correction_2d] WARNING: {n_below} events below "
            f"log10(E)={edges[0]:.3f} and {n_above} above {edges[-1]:.3f} "
            f"clamped to edge bins; consider widening the BC fit range.\n"
        )
    bins = np.clip(np.digitize(log_e, edges) - 1, 0, bf.shape[1] - 1)
    bf_per_event = bf[:, bins].T  # [N, nl]
    corrected = np.round(gen_counts[:, :nl].astype(np.float64) * bf_per_event)
    corrected = np.clip(corrected, 0, np.iinfo(np.int32).max).astype(np.int32)
    result = gen_counts.copy()
    result[:, :nl] = corrected
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--downstream", required=True, choices=list(DOWNSTREAMS.keys()))
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-bias-correction",
        action="store_true",
        default=False,
        help="Legacy alias for --bias-mode native.",
    )
    p.add_argument(
        "--bias-mode",
        choices=["1d", "2d", "native"],
        default="1d",
        help="1d (default): per-layer BC. 2d: per-(layer,E-bin) BC. native: none.",
    )
    p.add_argument("--scales", nargs="+", default=None)
    p.add_argument("--strategies", nargs="+", default=["finetune", "scratch"])
    p.add_argument(
        "--pcfm-base",
        default=None,
        help="Override PCFM root (absolute or relative to PCFM repo). "
        "Default: cfg['pcfm_root'].",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Override output dir (absolute path). Default: cfg['out_dir'].",
    )
    p.add_argument(
        "--runs-as-subdirs",
        action="store_true",
        default=False,
        help="Treat each scale as the model dir directly (skip strategy subdir join). "
        "Use for flat layouts like from_LEMURS_pretrain/{100,1k,10k,100k}/.",
    )
    p.add_argument(
        "--cond-file",
        type=str,
        default=None,
        help="Override DOWNSTREAMS[downstream]['cond_file']. Path to a cond .h5 "
        "with the same schema (energies, directions, num_layers, "
        "num_points_per_layer, sampling_fraction, pdg, layer_z_pos).",
    )
    p.add_argument(
        "--event-range",
        type=str,
        default=None,
        help='Half-open event-index slice "<start>:<end>" applied to cond_file. '
        "Default: all events.",
    )
    p.add_argument(
        "--bc-fit-mode",
        choices=["sequential", "random"],
        default="sequential",
        help="Pass-through to compute_bias_correction.py.",
    )
    p.add_argument(
        "--bc-fit-n",
        type=int,
        default=None,
        help="Pass-through to compute_bias_correction.py.",
    )
    p.add_argument(
        "--bc-fit-pool-end",
        type=int,
        default=None,
        help="Pass-through to compute_bias_correction.py.",
    )
    p.add_argument(
        "--bc-fit-seed",
        type=int,
        default=42,
        help="Pass-through to compute_bias_correction.py.",
    )
    p.add_argument(
        "--bc-clamp-max",
        type=float,
        default=2.0,
        help="Pass-through to compute_bias_correction.py: upper clamp "
        "for bias_factor (default 2.0).",
    )
    p.add_argument(
        "--sampling-temperature",
        type=float,
        default=1.0,
        help="Scale initial noise x0 by this factor (default 1.0). "
        "Lower → narrower output distribution, fewer outliers.",
    )
    p.add_argument(
        "--rejection-max-per-layer",
        type=int,
        default=0,
        help="Per-event rejection-sampling threshold on max post-BC count per "
        "active layer. 0 (default) disables rejection. Events whose "
        "max_per_layer >= T are regenerated up to --rejection-n-max times "
        "with a fresh RNG branch; the conditioning (E, theta, phi) is "
        "preserved. Recommended for LEMURS pipeline: T=5000 (matches "
        "AllShowers FlexAttention compile bound).",
    )
    p.add_argument(
        "--rejection-n-max",
        type=int,
        default=10,
        help="Maximum retries per event in rejection sampling (default 10). "
        "Events still failing after N_max retries are kept (caller may filter).",
    )
    args = p.parse_args()
    if args.no_bias_correction:
        args.bias_mode = "native"

    cfg = DOWNSTREAMS[args.downstream]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, downstream: {args.downstream}")

    cond_path = Path(args.cond_file) if args.cond_file else Path(cfg["cond_file"])
    if args.event_range:
        s, e = (int(x) for x in args.event_range.split(":"))
        ev_slice = slice(s, e)
        ev_tag = f"_evt{s}-{e}"
    else:
        ev_slice = slice(None)
        ev_tag = ""
    if args.pcfm_base is not None:
        pcfm_root = (
            Path(args.pcfm_base)
            if os.path.isabs(args.pcfm_base)
            else REPO / args.pcfm_base
        )
    else:
        pcfm_root = REPO / cfg["pcfm_root"]
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    nl_val = cfg["nl_val"]
    sf_val = cfg["sf_val"]
    n_active = cfg["n_active"]
    scales = args.scales or cfg["scales"]
    subdirs = cfg["subdirs"]

    # Load conditioning file
    print(f"\nReading cond file: {cond_path}")
    with h5py.File(cond_path, "r") as f:
        cond_nl = f["num_layers"][ev_slice].flatten()
        cond_npts = f["num_points_per_layer"][ev_slice]
        all_keys = list(f.keys())
        cond_data = {k: f[k][ev_slice] for k in all_keys}

    N = len(cond_nl)
    nl_mask = cond_nl == nl_val
    n_match = nl_mask.sum()
    print(
        f"  {N} events | {n_match} with nl={nl_val} (replace) | {N - n_match} keep G4"
    )

    E_match = cond_data["energies"][nl_mask].flatten()
    dir_match = cond_data["directions"][nl_mask]

    stem = cond_path.stem.replace("_cond", "")  # e.g. "simplebox_10k"

    # Loop over (scale, strategy) pairs
    for scale in scales:
        for strategy in args.strategies:
            if args.runs_as_subdirs:
                model_dir = pcfm_root / scale
            else:
                subdir = subdirs.get(strategy)
                if subdir is None:
                    continue
                model_dir = pcfm_root / scale / subdir
                # Fallback for finetune
                if (
                    not model_dir.exists()
                    and strategy == "finetune"
                    and "finetune_fallback" in subdirs
                ):
                    model_dir = pcfm_root / scale / subdirs["finetune_fallback"]

            out_name = f"{stem}_pcfm_D{scale}_{strategy}{ev_tag}.h5"
            out_path = out_dir / out_name

            print(f"\n{'─' * 60}")
            print(f"  {scale}/{strategy} → {out_name}")

            if not model_dir.exists():
                print(f"  [SKIP] {model_dir} not found")
                continue

            try:
                model, transform, norm_stats, config = load_pcfm_model(
                    model_dir, device
                )
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            if args.bias_mode != "native":
                extra = []
                if args.bc_fit_mode != "sequential":
                    extra += ["--bc-fit-mode", args.bc_fit_mode]
                    if args.bc_fit_n is not None:
                        extra += ["--bc-fit-n", str(args.bc_fit_n)]
                    if args.bc_fit_pool_end is not None:
                        extra += ["--bc-fit-pool-end", str(args.bc_fit_pool_end)]
                    extra += ["--bc-fit-seed", str(args.bc_fit_seed)]
                if args.bc_clamp_max != 2.0:
                    extra += ["--bc-clamp-max", str(args.bc_clamp_max)]
                refresh_bias_correction(
                    model_dir, mode=args.bias_mode, extra_args=extra
                )

            print(f"    Generating for {n_match} events ...")
            gen_npts = generate_num_points(
                model,
                transform,
                norm_stats,
                config,
                E_match,
                dir_match,
                nl_val,
                sf_val,
                device,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                seed=args.seed,
                temperature=args.sampling_temperature,
            )
            if args.sampling_temperature != 1.0:
                print(f"    Sampling temperature: {args.sampling_temperature}")

            # BC application — encapsulated so the rejection loop can reuse it
            bc_path = None
            if args.bias_mode == "1d":
                bc_path = model_dir / "bias_correction.npz"
                if not bc_path.exists():
                    bc_path = None
            elif args.bias_mode == "2d":
                bc_path = model_dir / "bias_correction_2d.npz"
                if not bc_path.exists():
                    raise FileNotFoundError(
                        f"--bias-mode=2d requested but {bc_path} not found. "
                        "compute_bias_correction.py --bias-mode 2d failed or did not run. "
                        "Refusing to silently emit native output mislabeled as 2D."
                    )

            def apply_bc(counts, energies):
                """Apply BC matching args.bias_mode. Identity if no BC configured."""
                if args.bias_mode == "1d" and bc_path is not None:
                    return apply_bias_correction(counts, bc_path)
                if args.bias_mode == "2d":
                    return apply_bias_correction_2d(counts, bc_path, energies)
                return counts

            gen_npts = apply_bc(gen_npts, E_match)
            if args.bias_mode != "native" and bc_path is not None:
                print(f"    Bias correction applied (mode={args.bias_mode}).")

            # Rejection sampling on post-BC max-per-layer
            if args.rejection_max_per_layer > 0:
                T_rej = args.rejection_max_per_layer
                N_max_rej = args.rejection_n_max
                over_mask = gen_npts[:, :n_active].max(axis=1) >= T_rej
                n_init_bad = int(over_mask.sum())
                print(
                    f"    Rejection: T={T_rej} N_max={N_max_rej} "
                    f"initial_bad={n_init_bad}/{n_match} "
                    f"({100 * n_init_bad / max(n_match, 1):.4f}%)"
                )
                retries = np.zeros(n_match, dtype=np.int32)
                for retry in range(1, N_max_rej + 1):
                    if not over_mask.any():
                        break
                    bad_idx = np.where(over_mask)[0]
                    retries[bad_idx] += 1
                    new_gen = generate_num_points(
                        model,
                        transform,
                        norm_stats,
                        config,
                        E_match[bad_idx],
                        dir_match[bad_idx] if dir_match is not None else None,
                        nl_val,
                        sf_val,
                        device,
                        n_steps=args.n_steps,
                        batch_size=args.batch_size,
                        seed=args.seed + 1_000_000 * retry,
                        temperature=args.sampling_temperature,
                    )
                    new_gen = apply_bc(new_gen, E_match[bad_idx])
                    # Replace the bad rows; new_gen has shape [len(bad_idx), dim_input]
                    gen_npts[bad_idx] = new_gen[:, : gen_npts.shape[1]]
                    over_mask = gen_npts[:, :n_active].max(axis=1) >= T_rej
                n_final_bad = int(over_mask.sum())
                pct_init = 100 * n_init_bad / max(n_match, 1)
                pct_final = 100 * n_final_bad / max(n_match, 1)
                print(
                    f"    Rejection done: final_bad={n_final_bad}/{n_match} "
                    f"({pct_final:.4f}%); reduced from {pct_init:.4f}%"
                )
                print(
                    f"    Retry stats: mean={retries.mean():.4f} "
                    f"max={int(retries.max())} touched_events={int((retries > 0).sum())}"
                )

            # Sanity check
            means_gen = gen_npts[:, :n_active].mean(axis=0)
            means_g4 = cond_npts[nl_mask, :n_active].mean(axis=0)
            print(
                f"    Mean total gen={means_gen.sum():.0f} vs G4={means_g4.sum():.0f}"
            )

            # Write output
            new_npts = cond_npts.copy()
            new_npts[nl_mask] = gen_npts[:, : cond_npts.shape[1]].astype(np.int32)

            with h5py.File(out_path, "w") as fout:
                for k in all_keys:
                    if k == "num_points_per_layer":
                        fout.create_dataset(k, data=new_npts)
                    else:
                        fout.create_dataset(k, data=cond_data[k])
                fout.attrs["pcfm_model_dir"] = str(model_dir)
                fout.attrs["pcfm_scale"] = scale
                fout.attrs["pcfm_strategy"] = strategy
                fout.attrs["source_cond_file"] = str(cond_path)
                fout.attrs["n_steps"] = args.n_steps
                fout.attrs["bias_correction"] = args.bias_mode != "native"
                fout.attrs["bias_mode"] = args.bias_mode
                fout.attrs["rejection_max_per_layer"] = args.rejection_max_per_layer
                fout.attrs["rejection_n_max"] = args.rejection_n_max

            print(f"    Saved → {out_path}")

    print(f"\n{'=' * 60}")
    print(f"Done. Files in: {out_dir}")


if __name__ == "__main__":
    main()
