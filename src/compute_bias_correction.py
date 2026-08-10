"""compute_bias_correction.py

Compute per-layer multiplicative bias correction factors for a trained
PointCountFM ALLEGRO model and evaluate the corrected physics score.

The Jensen's inequality bias (exp() amplifies z-space residual noise → positive
per-layer count bias) is fit from a large sample and saved as a numpy array.

Usage
-----
    cd src/
    python compute_bias_correction.py \
        --result_dir ../results/ALLEGRO/100k/from_scratch \
        --data /path/to/allegro_100k_pcfm.h5 \
        --n_samples 5000

Output
------
    <result_dir>/bias_correction.npz
        bias_factor         float64 [nl]   — multiply gen[:, :nl] by this
        real_layer_means    float64 [nl]
        gen_layer_means     float64 [nl]
        active_layer_mask   bool    [nl]   — layers with real_mean > 0.5
        n_layers            int            — nl used for correction
        n_samples           int
        n_bins              int
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

# allow running from src/ or project root
sys.path.insert(0, str(Path(__file__).parent))

from model import build_model_from_config
from pcfm_utils import (
    create_transforms,
    extract_scaler_scalar,
    heun_sampler,
    inverse_transform,
)
from preprocessing import detect_active_dims


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-layer bias correction for a trained ALLEGRO PointCountFM model."
    )
    p.add_argument(
        "--result_dir",
        required=True,
        help="Path to result directory (contains best_physics_model.pt and conf.yaml)",
    )
    p.add_argument(
        "--data", default=None, help="HDF5 data file path (overrides config)"
    )
    p.add_argument(
        "--checkpoint",
        default="best_physics_model.pt",
        help="Checkpoint filename to load (default: best_physics_model.pt)",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="Samples per energy bin for bias estimation (default: 5000)",
    )
    p.add_argument(
        "--n_steps", type=int, default=200, help="ODE integration steps (default: 200)"
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output .npz path (default: <result_dir>/bias_correction[_2d].npz)",
    )
    p.add_argument(
        "--bias-mode",
        choices=["1d", "2d"],
        default="1d",
        help="1d (default): per-layer multiplicative BC. 2d: per-(layer,energy-bin) BC.",
    )
    p.add_argument(
        "--n-energy-bins",
        type=int,
        default=8,
        help="Number of log10(E) bins for 2D mode (default: 8).",
    )
    p.add_argument(
        "--min-events-per-cell",
        type=int,
        default=50,
        help="2D fit fallback threshold per (layer,bin) cell (default: 50).",
    )
    p.add_argument(
        "--bc-fit-mode",
        choices=["sequential", "random"],
        default="sequential",
        help="sequential (default): use train[:n_train] for BC fit (legacy). "
        "random: sample bc-fit-n indices from train[:bc-fit-pool-end].",
    )
    p.add_argument(
        "--bc-fit-n",
        type=int,
        default=None,
        help="Random mode: number of events to sample.",
    )
    p.add_argument(
        "--bc-fit-pool-end",
        type=int,
        default=None,
        help="Random mode: sample from train[0:bc-fit-pool-end].",
    )
    p.add_argument(
        "--bc-fit-seed",
        type=int,
        default=42,
        help="Random mode: seed for np.random.default_rng (default: 42).",
    )
    p.add_argument(
        "--bc-clamp-max",
        type=float,
        default=2.0,
        help="Upper clamp for bias_factor (default: 2.0). "
        "Lower clamp stays 0.5. Applied to per-layer 1D BC, "
        "per-bin total BC, and 2D BC.",
    )
    return p.parse_args()


def _build_conditions(energies, e_mean, e_std, norm_sf, norm_nl, use_dir, dirs, device):
    """Build normalized condition tensor, matching trainer.__build_conditions."""
    n = len(energies)
    log_e = (np.log(energies) - e_mean) / e_std
    cols = [log_e, np.full(n, norm_sf), np.full(n, norm_nl)]

    if use_dir:
        if dirs is None:
            dirs = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
        dirs = np.asarray(dirs, dtype=np.float32)
        if dirs.ndim == 1:
            dirs = np.tile(dirs, (n, 1))
        cols.append(dirs)

    return torch.FloatTensor(np.column_stack(cols)).to(device)


def compute_physics_score(gen_dict, real_dict, energy_bins, nl):
    """mean |gen_layer/real_layer - 1| across active layers and energy bins."""
    scores = []
    for ebin in energy_bins:
        if ebin not in gen_dict:
            continue
        gen_profile = gen_dict[ebin][:, :nl].mean(axis=0)
        real_profile = real_dict[ebin][:, :nl].mean(axis=0)
        mask = real_profile > 0.5
        if mask.any():
            ratios = gen_profile[mask] / real_profile[mask]
            scores.append(np.mean(np.abs(ratios - 1)))
    return np.mean(scores) * 100 if scores else None


def apply_correction(gen_dict, bias_factor, nl, dim_input):
    """Apply multiplicative bias correction in count space."""
    corrected = {}
    for ebin, gen in gen_dict.items():
        gen_float = gen[:, :nl].astype(np.float64) * bias_factor[np.newaxis, :]
        gen_corr = np.clip(np.round(gen_float), 0, np.iinfo(np.int32).max).astype(
            np.int32
        )
        padded = np.zeros((gen.shape[0], dim_input), dtype=np.int32)
        padded[:, :nl] = gen_corr
        corrected[ebin] = padded
    return corrected


def _eval_val_physics_score(
    model,
    transform_num_points,
    active_mask,
    dim_input,
    nl,
    sf,
    e_mean,
    e_std,
    norm_sf,
    norm_nl,
    use_dir,
    val_data,
    val_energy,
    val_nlayers,
    val_directions,
    n_steps,
    n_per_bin,
    device,
    bias_factor,
    bias_factor_2d,
    e_bin_edges,
    active_layer_mask,
    do_2d,
):
    """Generate val-side samples and compute physics_score under {native,1D,2D}.

    Returns list of (label, score%) tuples.
    """
    if val_nlayers is None or len(val_energy) == 0:
        return [("(val unavailable)", None)]
    nl_mask = val_nlayers == nl
    val_energy_nl = val_energy[nl_mask]
    val_data_nl = val_data[nl_mask]
    val_dirs_nl = val_directions[nl_mask] if val_directions is not None else None
    if len(val_energy_nl) == 0:
        return [("(no val events at nl=max)", None)]

    # Same percentile-bin protocol as fit, but on val side
    percentiles = np.percentile(val_energy_nl, np.linspace(0, 100, 6))
    val_bins = [(float(percentiles[i]), float(percentiles[i + 1])) for i in range(5)]

    val_gen_dict = {}
    val_real_dict = {}
    val_e_dict = {}
    with torch.no_grad():
        for idx, ebin in enumerate(val_bins):
            e_lo, e_hi = ebin
            pool = np.where((val_energy_nl >= e_lo) & (val_energy_nl <= e_hi))[0]
            if len(pool) == 0:
                continue
            np.random.seed(13 + idx)
            chosen = np.random.choice(pool, size=n_per_bin, replace=True)
            energies = val_energy_nl[chosen]
            dirs = val_dirs_nl[chosen] if val_dirs_nl is not None else None
            cond = _build_conditions(
                energies, e_mean, e_std, norm_sf, norm_nl, use_dir, dirs, device
            )
            torch.manual_seed(13 + idx)
            x0 = torch.randn(n_per_bin, dim_input, device=device)
            if active_mask is not None:
                x0 = x0 * active_mask
            gen_raw = heun_sampler(model, x0, cond, n_steps)
            gen_counts = inverse_transform(gen_raw, transform_num_points)
            val_gen_dict[ebin] = gen_counts
            val_real_dict[ebin] = val_data_nl[chosen]
            val_e_dict[ebin] = energies

    rows = []
    # native
    s_nat = compute_physics_score(val_gen_dict, val_real_dict, val_bins, nl)
    rows.append(("native", s_nat))
    # 1D BC
    val_gen_1d = apply_correction(val_gen_dict, bias_factor, nl, dim_input)
    s_1d = compute_physics_score(val_gen_1d, val_real_dict, val_bins, nl)
    rows.append(("1D BC", s_1d))
    # 2D BC (only if computed)
    if do_2d and bias_factor_2d is not None and e_bin_edges is not None:
        val_gen_2d = {}
        for eb, gen in val_gen_dict.items():
            energies = val_e_dict[eb]
            log_e = np.log10(np.asarray(energies, dtype=np.float64))
            bins = np.clip(
                np.digitize(log_e, e_bin_edges) - 1, 0, bias_factor_2d.shape[1] - 1
            )
            bf_per_event = bias_factor_2d[:, bins].T  # [N, nl]
            gen_float = gen[:, :nl].astype(np.float64) * bf_per_event
            corr = np.clip(np.round(gen_float), 0, np.iinfo(np.int32).max).astype(
                np.int32
            )
            padded = np.zeros((gen.shape[0], dim_input), dtype=np.int32)
            padded[:, :nl] = corr
            val_gen_2d[eb] = padded
        s_2d = compute_physics_score(val_gen_2d, val_real_dict, val_bins, nl)
        rows.append(("2D BC", s_2d))
    return rows


def main():
    args = _parse_args()
    result_dir = Path(args.result_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load config ────────────────────────────────────────────────────────────
    with open(result_dir / "conf.yaml") as f:
        config = yaml.safe_load(f)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = result_dir / args.checkpoint
    if not ckpt_path.exists():
        ckpt_path = result_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {result_dir}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    norm_stats = checkpoint["norm_stats"]
    ep = checkpoint.get("epoch", "?")
    vl = checkpoint.get("val_loss")
    if vl is not None:
        print(f"  Epoch {ep}, val_loss={vl:.4f}")
    else:
        print(f"  Epoch {ep}  (val_loss n/a — staged/snapshot ckpt)")
    if "physics_score" in checkpoint:
        print(f"  Saved physics score: {checkpoint['physics_score']:.2f}%")

    # ── Create transforms ──────────────────────────────────────────────────────
    transform_num_points, transform_energy, transform_nlayers = create_transforms(
        config, norm_stats
    )

    # ── Build model ────────────────────────────────────────────────────────────
    mc = config["model"]
    dim_input = mc["dim_input"]
    model = build_model_from_config(config, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Detect active dims ─────────────────────────────────────────────────────
    active_mask = detect_active_dims(transform_num_points, device)
    if active_mask is not None:
        n_active = int(active_mask.sum().item())
        print(f"  Active dims: {n_active}/{dim_input}")

    # ── Load data (training split only — no validation leakage) ───────────────
    data_file = args.data or config["data"]["data_file"]
    use_dir = config["data"].get("use_direction_conditioning", False)
    use_nlayers = config["data"].get("use_nlayers_conditioning", False)

    # Replicate the exact train/val split from dataset.get_dataloaders
    d_cfg = config["data"]
    val_samples = d_cfg.get("val_samples", None)
    max_samples = d_cfg.get("max_samples", None)
    train_fraction = d_cfg.get("train_fraction", 0.9)

    with h5py.File(data_file, "r") as f:
        total_samples = f["energy"].shape[0]
        if val_samples is not None:
            n_train = (
                min(max_samples, total_samples - val_samples)
                if max_samples is not None
                else max(0, total_samples - val_samples)
            )
        else:
            n_total = (
                min(total_samples, max_samples)
                if max_samples is not None
                else total_samples
            )
            n_train = int(n_total * train_fraction)

        print(f"Loading data: {data_file}")
        print(
            f"  Split: {n_train} train / {total_samples - n_train} val — using train only"
        )
        if args.bc_fit_mode == "random":
            if args.bc_fit_n is None or args.bc_fit_pool_end is None:
                raise ValueError(
                    "--bc-fit-mode random requires --bc-fit-n and --bc-fit-pool-end"
                )
            if args.bc_fit_pool_end > n_train:
                raise ValueError(
                    f"--bc-fit-pool-end={args.bc_fit_pool_end} > n_train={n_train}; "
                    "BC fit pool would overlap val partition (leakage)."
                )
            rng = np.random.default_rng(args.bc_fit_seed)
            fit_idx = np.sort(
                rng.choice(args.bc_fit_pool_end, args.bc_fit_n, replace=False)
            )
            print(
                f"  BC fit: random N={args.bc_fit_n} from train[0:{args.bc_fit_pool_end}], seed={args.bc_fit_seed}"
            )
            real_data = f["num_points"][:][fit_idx]
            real_energy = f["energy"][:][fit_idx].flatten()
            real_fsamp = f["sampling_fraction"][:][fit_idx].flatten()
            real_nlayers = (
                f["n_layers"][:][fit_idx].flatten() if "n_layers" in f else None
            )
            real_directions = (
                f["directions"][:][fit_idx] if (use_dir and "directions" in f) else None
            )
        else:
            real_data = f["num_points"][:n_train]
            real_energy = f["energy"][:n_train].flatten()
            real_fsamp = f["sampling_fraction"][:n_train].flatten()
            real_nlayers = (
                f["n_layers"][:n_train].flatten() if "n_layers" in f else None
            )
            real_directions = (
                f["directions"][:n_train] if (use_dir and "directions" in f) else None
            )
        # ── R1 held-out val slice (no leakage into BC fit; used only for scoring) ─
        val_data = f["num_points"][n_train:]
        val_energy = f["energy"][n_train:].flatten()
        val_nlayers = f["n_layers"][n_train:].flatten() if "n_layers" in f else None
        val_directions = (
            f["directions"][n_train:] if (use_dir and "directions" in f) else None
        )
    n_val = len(val_energy)
    print(f"  Loaded {n_train} training events, {n_val} held-out val events")

    if real_nlayers is None:
        raise ValueError("n_layers not found in data file — required for Allegro eval")

    # ── Allegro eval protocol: fixed nl=max, sf=median, 5 energy percentile bins
    nl = int(np.max(real_nlayers))
    sf = float(np.median(real_fsamp))
    print(f"  Eval protocol: nl={nl}, sf={sf:.4f}")

    nl_mask = real_nlayers == nl
    real_energy_nl = real_energy[nl_mask]
    real_data_nl = real_data[nl_mask]
    real_dirs_nl = real_directions[nl_mask] if real_directions is not None else None

    percentiles = np.percentile(real_energy_nl, np.linspace(0, 100, 6))
    energy_bins = [(float(percentiles[i]), float(percentiles[i + 1])) for i in range(5)]
    print(
        "  Energy bins: " + ", ".join(f"[{lo:.1f},{hi:.1f}]" for lo, hi in energy_bins)
    )

    # ── Normalize scalar conditions ─────────────────────────────────────────────
    e_mean, e_std = extract_scaler_scalar(transform_energy)

    # fsamp normalization
    use_minmax_fsamp = "fsamp_data_min" in norm_stats
    if use_minmax_fsamp:
        f_min = float(np.atleast_1d(norm_stats["fsamp_data_min"]).item())
        f_max = float(np.atleast_1d(norm_stats["fsamp_data_max"]).item())
        f_tmin = norm_stats["fsamp_target_min"]
        f_tmax = norm_stats["fsamp_target_max"]
        if f_max == f_min:
            norm_sf = (f_tmin + f_tmax) / 2.0  # constant fsamp → center of range
        else:
            norm_sf = (sf - f_min) / (f_max - f_min) * (f_tmax - f_tmin) + f_tmin
    else:
        f_mean = float(np.atleast_1d(norm_stats["fsamp_mean"]).item())
        f_std = float(np.atleast_1d(norm_stats["fsamp_std"]).item())
        norm_sf = (sf - f_mean) / f_std

    # n_layers normalization
    if use_nlayers and transform_nlayers is not None:
        norm_nl = transform_nlayers(torch.tensor([[nl]], dtype=torch.float32)).item()
    else:
        norm_nl = float(nl)

    # ── Generate ───────────────────────────────────────────────────────────────
    n_samples = args.n_samples
    n_steps = args.n_steps
    gen_dict = {}
    real_dict = {}
    energy_dict = {}  # per-event energies (GeV) for the events used in the fit

    print(f"\nGenerating {n_samples} samples/bin × {len(energy_bins)} bins ...")
    with torch.no_grad():
        for idx, ebin in enumerate(energy_bins):
            e_lo, e_hi = ebin
            pool = np.where((real_energy_nl >= e_lo) & (real_energy_nl <= e_hi))[0]
            if len(pool) == 0:
                print(f"  Bin {idx + 1}: no real samples, skipping")
                continue

            np.random.seed(42 + idx)
            chosen = np.random.choice(pool, size=n_samples, replace=True)
            energies = real_energy_nl[chosen]
            dirs = real_dirs_nl[chosen] if real_dirs_nl is not None else None

            cond = _build_conditions(
                energies, e_mean, e_std, norm_sf, norm_nl, use_dir, dirs, device
            )
            torch.manual_seed(42 + idx)
            x0 = torch.randn(n_samples, dim_input, device=device)
            if active_mask is not None:
                x0 = x0 * active_mask

            gen_raw = heun_sampler(model, x0, cond, n_steps)
            gen_counts = inverse_transform(gen_raw, transform_num_points)

            gen_dict[ebin] = gen_counts
            real_dict[ebin] = real_data_nl[chosen]
            energy_dict[ebin] = energies

            gen_total = gen_counts[:, :nl].sum(1)
            real_total = real_data_nl[chosen][:, :nl].sum(1)
            bias_pct = (
                (gen_total.mean() - real_total.mean()) / real_total.mean() * 100
                if real_total.mean() > 0
                else float("nan")
            )
            print(
                f"  Bin {idx + 1} [{e_lo:.1f},{e_hi:.1f} GeV]: "
                f"Gen={gen_total.mean():.0f}±{gen_total.std():.0f}, "
                f"Real={real_total.mean():.0f}±{real_total.std():.0f}, "
                f"TotalBias={bias_pct:+.1f}%"
            )

    # ── Compute per-layer bias factors ─────────────────────────────────────────
    print(
        f"\nComputing per-layer bias factors (averaged over {len(gen_dict)} bins) ..."
    )
    gen_layer_sum = np.zeros(nl, dtype=np.float64)
    real_layer_sum = np.zeros(nl, dtype=np.float64)
    n_valid_bins = 0

    for ebin in energy_bins:
        if ebin not in gen_dict:
            continue
        gen_layer_sum += gen_dict[ebin][:, :nl].mean(axis=0)
        real_layer_sum += real_dict[ebin][:, :nl].mean(axis=0)
        n_valid_bins += 1

    gen_layer_means = gen_layer_sum / n_valid_bins
    real_layer_means = real_layer_sum / n_valid_bins

    # bias_factor[l] = real_mean[l] / gen_mean[l]  (< 1 iff model overshoots)
    active_layer_mask = real_layer_means > 0.5  # skip near-zero (dead) layers
    bias_factor = np.ones(nl, dtype=np.float64)
    bias_factor[active_layer_mask] = real_layer_means[
        active_layer_mask
    ] / gen_layer_means[active_layer_mask].clip(1e-3)
    # Clamp to [0.5, bc_clamp_max] — guard against degenerate estimates
    bias_factor = np.clip(bias_factor, 0.5, args.bc_clamp_max)
    print(f"  Clamp window: [0.5, {args.bc_clamp_max}]")

    # ── Per-layer report ───────────────────────────────────────────────────────
    print(
        f"\n{'Layer':>6}  {'Real mean':>10}  {'Gen mean':>10}  "
        f"{'Bias':>8}  {'Correction':>10}"
    )
    print("-" * 56)
    for layer_idx in range(nl):
        if not active_layer_mask[layer_idx]:
            continue
        bias_pct = (
            (gen_layer_means[layer_idx] - real_layer_means[layer_idx])
            / max(real_layer_means[layer_idx], 0.5)
            * 100
        )
        print(
            f"  L{layer_idx:02d}   {real_layer_means[layer_idx]:10.2f}  {gen_layer_means[layer_idx]:10.2f}  "
            f"{bias_pct:+8.1f}%   x{bias_factor[layer_idx]:.4f}"
        )

    # ── Physics score before correction ───────────────────────────────────────
    score_before = compute_physics_score(gen_dict, real_dict, energy_bins, nl)
    print(f"\nPhysics score BEFORE correction: {score_before:.2f}%")

    # ── Physics score after correction ────────────────────────────────────────
    gen_dict_corr = apply_correction(gen_dict, bias_factor, nl, dim_input)
    score_after = compute_physics_score(gen_dict_corr, real_dict, energy_bins, nl)
    improvement = score_before - score_after
    rel_improvement = improvement / score_before * 100
    print(f"Physics score AFTER  correction: {score_after:.2f}%")
    print(
        f"Improvement:                     {improvement:.2f}pp "
        f"({rel_improvement:.1f}% relative)"
    )

    # ── Per-bin total correction (residual after per-layer BC) ─────────────────
    # After per-layer BC, a residual energy-dependent scale offset can remain
    # because per-layer factors are averaged across bins.  A per-bin global
    # multiplier corrects the total-hits mean independently for each energy bin.
    valid_ebins = [eb for eb in energy_bins if eb in gen_dict_corr]
    n_valid = len(valid_ebins)
    total_bc_per_bin = np.ones(n_valid, dtype=np.float64)
    energy_bin_lo = np.array([lo for lo, hi in valid_ebins], dtype=np.float64)
    energy_bin_hi = np.array([hi for lo, hi in valid_ebins], dtype=np.float64)

    print("\nPer-bin total correction (after per-layer BC):")
    print(
        f"{'Bin':>4}  {'Real total':>10}  {'Gen total':>10}  {'Residual':>9}  {'Factor':>8}"
    )
    print("-" * 52)
    for i, ebin in enumerate(valid_ebins):
        gen_total = gen_dict_corr[ebin][:, :nl].sum(axis=1).mean()
        real_total = real_dict[ebin][:, :nl].sum(axis=1).mean()
        if gen_total > 1e-3 and real_total > 1e-3:
            total_bc_per_bin[i] = real_total / gen_total
        total_bc_per_bin[i] = float(
            np.clip(total_bc_per_bin[i], 0.5, args.bc_clamp_max)
        )
        residual_pct = (gen_total / max(real_total, 1e-3) - 1.0) * 100
        print(
            f"  {i + 1:2d}  {real_total:10.0f}  {gen_total:10.0f}  "
            f"{residual_pct:+8.1f}%   x{total_bc_per_bin[i]:.4f}"
        )

    # ── 2D fit (only when --bias-mode 2d) ─────────────────────────────────────
    bias_factor_2d = None
    e_bin_edges = None
    n_events_per_cell = None
    fallback_layer_mask = None
    if args.bias_mode == "2d":
        N_BINS = args.n_energy_bins
        MIN_EVENTS = args.min_events_per_cell
        all_gen = np.concatenate(
            [gen_dict[eb][:, :nl] for eb in energy_bins if eb in gen_dict], axis=0
        ).astype(np.float64)
        all_real = np.concatenate(
            [real_dict[eb][:, :nl] for eb in energy_bins if eb in real_dict], axis=0
        ).astype(np.float64)
        all_e = np.concatenate(
            [energy_dict[eb] for eb in energy_bins if eb in energy_dict]
        ).astype(np.float64)
        log_e = np.log10(all_e)
        e_bin_edges = np.linspace(log_e.min(), log_e.max(), N_BINS + 1)
        # digitize returns 1..N_BINS for in-range values; map to 0..N_BINS-1
        bin_idx = np.clip(np.digitize(log_e, e_bin_edges) - 1, 0, N_BINS - 1)

        bias_factor_2d = np.ones((nl, N_BINS), dtype=np.float64)
        n_events_per_cell = np.zeros((nl, N_BINS), dtype=np.int64)
        fallback_layer_mask = np.zeros((nl, N_BINS), dtype=bool)

        for b in range(N_BINS):
            mask_b = bin_idx == b
            n_b = int(mask_b.sum())
            for layer_idx in range(nl):
                n_events_per_cell[layer_idx, b] = n_b
                if n_b < MIN_EVENTS:
                    bias_factor_2d[layer_idx, b] = bias_factor[
                        layer_idx
                    ]  # fallback to 1D for this layer
                    fallback_layer_mask[layer_idx, b] = True
                    continue
                g4_lb = float(all_real[mask_b, layer_idx].mean())
                gen_lb = float(all_gen[mask_b, layer_idx].mean())
                if g4_lb < 0.5 or gen_lb < 1e-3:
                    bias_factor_2d[layer_idx, b] = bias_factor[layer_idx]
                    fallback_layer_mask[layer_idx, b] = True
                    continue
                bias_factor_2d[layer_idx, b] = g4_lb / gen_lb
        bias_factor_2d = np.clip(bias_factor_2d, 0.5, args.bc_clamp_max)

        # ── R2: per-(E_bin) total-hits closure under 2D BC ──────────────
        print(f"\n2D BC per-bin total-hits closure ({N_BINS} log10(E) bins):")
        print(
            f"{'bin':>4}  {'log10(E) range':>20}  {'n':>6}  "
            f"{'tot_g4':>10}  {'tot_corr':>10}  {'residual':>9}"
        )
        print("-" * 72)
        for b in range(N_BINS):
            mask_b = bin_idx == b
            n_b = int(mask_b.sum())
            if n_b == 0:
                print(
                    f"{b:>4}  [{e_bin_edges[b]:>+.3f},{e_bin_edges[b + 1]:>+.3f}]   "
                    f"{n_b:>6}    (no events in bin)"
                )
                continue
            real_b = all_real[mask_b].astype(np.float64)
            gen_b = all_gen[mask_b].astype(np.float64)
            corr_b = gen_b * bias_factor_2d[:, b][np.newaxis, :]
            tot_g4 = float(real_b.sum(axis=1).mean())
            tot_corr = float(corr_b.sum(axis=1).mean())
            residual = (tot_corr / tot_g4 - 1.0) * 100 if tot_g4 > 0 else float("nan")
            print(
                f"{b:>4}  [{e_bin_edges[b]:>+.3f},{e_bin_edges[b + 1]:>+.3f}]   "
                f"{n_b:>6}  {tot_g4:>10.0f}  {tot_corr:>10.0f}  {residual:>+8.2f}%"
            )

        # ── R4: clip rate excluding fallback cells ──────────────────────
        clip_mask = (bias_factor_2d == 0.5) | (bias_factor_2d == 2.0)
        clip_genuine = clip_mask & ~fallback_layer_mask
        n_clipped_genuine = int(clip_genuine.sum())
        n_fallback = int(fallback_layer_mask.sum())
        n_non_fallback = nl * N_BINS - n_fallback
        clip_rate = n_clipped_genuine / n_non_fallback if n_non_fallback > 0 else 0.0
        print(
            f"\n2D BC: N_BINS={N_BINS}, log10(E) range "
            f"[{e_bin_edges[0]:.3f}, {e_bin_edges[-1]:.3f}], "
            f"fallback cells={n_fallback}/{nl * N_BINS}, "
            f"clipped (non-fallback)={n_clipped_genuine}/{n_non_fallback} "
            f"({clip_rate * 100:.1f}%)"
        )

    # ── R1: held-out val-slice physics score under {native, 1D, 2D} ──────────
    print(
        f"\nVal-slice physics score (held-out {n_val} events; "
        f"per-event mean |gen_layer/real_layer - 1| over active layers, "
        f"averaged across percentile bins):"
    )
    print(f"{'BC variant':>12}  {'physics_score (%)':>20}")
    print("-" * 36)
    val_score_rows = _eval_val_physics_score(
        model=model,
        transform_num_points=transform_num_points,
        active_mask=active_mask,
        dim_input=dim_input,
        nl=nl,
        sf=sf,
        e_mean=e_mean,
        e_std=e_std,
        norm_sf=norm_sf,
        norm_nl=norm_nl,
        use_dir=use_dir,
        val_data=val_data,
        val_energy=val_energy,
        val_nlayers=val_nlayers,
        val_directions=val_directions,
        n_steps=n_steps,
        n_per_bin=min(2000, n_samples),
        device=device,
        bias_factor=bias_factor,
        bias_factor_2d=bias_factor_2d,
        e_bin_edges=e_bin_edges,
        active_layer_mask=active_layer_mask,
        do_2d=(args.bias_mode == "2d"),
    )
    for label, score in val_score_rows:
        score_str = f"{score:.2f}" if score is not None else "n/a"
        print(f"{label:>12}  {score_str:>20}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.bias_mode == "1d":
        output_path = (
            Path(args.output) if args.output else result_dir / "bias_correction.npz"
        )
        np.savez(
            output_path,
            bias_factor=bias_factor,
            real_layer_means=real_layer_means,
            gen_layer_means=gen_layer_means,
            active_layer_mask=active_layer_mask,
            n_layers=nl,
            n_samples=n_samples,
            n_bins=n_valid_bins,
            total_bc_per_bin=total_bc_per_bin,
            energy_bin_lo=energy_bin_lo,
            energy_bin_hi=energy_bin_hi,
        )
        print(f"\nSaved (mode=1d): {output_path}")
        print(
            "  Keys: bias_factor, real_layer_means, gen_layer_means, active_layer_mask,"
        )
        print("        n_layers, n_samples, n_bins, total_bc_per_bin, energy_bin_lo,")
        print("        energy_bin_hi")
    else:
        output_path = (
            Path(args.output) if args.output else result_dir / "bias_correction_2d.npz"
        )
        np.savez(
            output_path,
            bias_factor_2d=bias_factor_2d,
            e_bin_edges=e_bin_edges,
            n_events_per_cell=n_events_per_cell,
            fallback_layer_mask=fallback_layer_mask,
            bias_factor_1d=bias_factor,  # keep 1D for downstream sanity
            active_layer_mask=active_layer_mask,
            n_layers=nl,
            n_samples=n_samples,
            n_bins=args.n_energy_bins,
            bias_mode="2d",
        )
        print(f"\nSaved (mode=2d): {output_path}")
        print("  Keys: bias_factor_2d [nl, N_BINS], e_bin_edges [N_BINS+1],")
        print(
            "        n_events_per_cell [nl, N_BINS], fallback_layer_mask [nl, N_BINS],"
        )
        print(
            "        bias_factor_1d [nl], active_layer_mask [nl], n_layers, n_samples,"
        )
        print("        n_bins (=N_BINS), bias_mode")
    print("\nTo apply at inference (1d):")
    print("  bc = np.load('bias_correction.npz')")
    print("  gen_corr = np.round(gen[:, :nl] * bc['bias_factor']).astype(np.int32)")
    print("To apply at inference (2d):")
    print("  bc = np.load('bias_correction_2d.npz')")
    print("  b = np.clip(np.digitize(np.log10(E_gev), bc['e_bin_edges']) - 1, 0, N-1)")
    print(
        "  gen_corr = np.round(gen[:, :nl] * bc['bias_factor_2d'][:, b].T).astype(np.int32)"
    )


if __name__ == "__main__":
    main()
