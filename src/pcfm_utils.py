"""Shared PCFM inference/transform utilities.

Extracted from the internal evaluation script so that training
(trainer.py), condition-file generation (generate_pcfm_cond.py) and the
bias-correction fit (compute_bias_correction.py) do not depend on the
plotting stack.
"""

import numpy as np
import torch


# Utility functions for transform stats extraction
def extract_scaler_stats(transform):
    """Extract mean/std from StandardScaler in transform pipeline."""
    if hasattr(transform, "sub_modules"):
        for module in transform.sub_modules:
            if "Standard" in module.__class__.__name__:
                return {
                    "mean": module.mean.cpu().numpy(),
                    "std": module.std.cpu().numpy(),
                }
    if "Standard" in transform.__class__.__name__:
        return {
            "mean": transform.mean.cpu().numpy(),
            "std": transform.std.cpu().numpy(),
        }
    return None


def extract_minmax_stats(transform):
    """Extract data_min/data_max from MinMaxScaler in transform pipeline."""
    if hasattr(transform, "sub_modules"):
        for module in transform.sub_modules:
            if "MinMax" in module.__class__.__name__:
                return {
                    "data_min": module.data_min.cpu().numpy(),
                    "data_max": module.data_max.cpu().numpy(),
                    "target_min": module.target_min,
                    "target_max": module.target_max,
                }
    if "MinMax" in transform.__class__.__name__:
        return {
            "data_min": transform.data_min.cpu().numpy(),
            "data_max": transform.data_max.cpu().numpy(),
            "target_min": transform.target_min,
            "target_max": transform.target_max,
        }
    return None


def extract_scaler_scalar(transform):
    """Extract mean and std as scalars from StandardScaler."""
    stats = extract_scaler_stats(transform)
    if stats:
        return stats["mean"].item(), stats["std"].item()
    return 0.0, 1.0


def create_norm_stats_dict(
    energy_transform, fsamp_transform, numpts_transform, nlayers_transform=None
):
    """Create normalization stats dictionary from transform objects."""
    energy_stats = extract_scaler_stats(energy_transform)
    numpts_stats = extract_scaler_stats(numpts_transform)

    stats = {
        "energy_mean": energy_stats["mean"] if energy_stats else 0.0,
        "energy_std": energy_stats["std"] if energy_stats else 1.0,
        "numpts_mean": numpts_stats["mean"] if numpts_stats else 0.0,
        "numpts_std": numpts_stats["std"] if numpts_stats else 1.0,
    }

    # Add fsamp stats (now uses MinMaxScaler)
    fsamp_stats = extract_minmax_stats(fsamp_transform)
    if fsamp_stats:
        stats["fsamp_data_min"] = fsamp_stats["data_min"]
        stats["fsamp_data_max"] = fsamp_stats["data_max"]
        stats["fsamp_target_min"] = fsamp_stats["target_min"]
        stats["fsamp_target_max"] = fsamp_stats["target_max"]
    else:
        # Fallback for old models using StandardScaler
        fsamp_std_stats = extract_scaler_stats(fsamp_transform)
        if fsamp_std_stats:
            stats["fsamp_mean"] = fsamp_std_stats["mean"]
            stats["fsamp_std"] = fsamp_std_stats["std"]

    # Add n_layers stats if provided (uses MinMaxScaler)
    if nlayers_transform is not None:
        nlayers_stats = extract_minmax_stats(nlayers_transform)
        if nlayers_stats:
            stats["nlayers_data_min"] = nlayers_stats["data_min"]
            stats["nlayers_data_max"] = nlayers_stats["data_max"]
            stats["nlayers_target_min"] = nlayers_stats["target_min"]
            stats["nlayers_target_max"] = nlayers_stats["target_max"]

    return stats


def create_transforms(config, norm_stats):
    """Create and configure transform pipelines from checkpoint stats."""
    from preprocessing import compose

    dim_input = config["model"]["dim_input"]

    transform_configs = {
        "num_points": config["data"].get(
            "transform_num_points",
            [["Log", {"alpha": 1.0}], ["StandardScaler", {"shape": [1, dim_input]}]],
        ),
        "energy": config["data"].get(
            "transform_inc",
            [["Log", {"alpha": 1e-8}], ["StandardScaler", {"shape": [1, 1]}]],
        ),
    }

    transforms = {k: compose(v) for k, v in transform_configs.items()}

    # Set normalization stats from checkpoint
    stats_map = {
        "num_points": ("numpts_mean", "numpts_std"),
        "energy": ("energy_mean", "energy_std"),
    }

    for name, (mean_key, std_key) in stats_map.items():
        for module in transforms[name].sub_modules:
            if "Standard" in module.__class__.__name__:
                mean = np.atleast_1d(np.array(norm_stats[mean_key]))
                std = np.atleast_1d(np.array(norm_stats[std_key]))
                module.mean = torch.from_numpy(mean).float().reshape(1, -1)
                module.std = torch.from_numpy(std).float().reshape(1, -1)
                break

    # Create n_layers transform if present in config
    nlayers_transform = None
    if config["data"].get("use_nlayers_conditioning", False):
        nlayers_config = config["data"].get(
            "transform_nlayers",
            [
                [
                    "MinMaxScaler",
                    {"shape": [1, 1], "target_min": -1.0, "target_max": 1.0},
                ]
            ],
        )
        nlayers_transform = compose(nlayers_config)

        # Set MinMaxScaler stats from checkpoint
        if "nlayers_data_min" in norm_stats:
            for module in nlayers_transform.sub_modules:
                if "MinMax" in module.__class__.__name__:
                    module.data_min = (
                        torch.tensor(norm_stats["nlayers_data_min"])
                        .float()
                        .reshape(1, -1)
                    )
                    module.data_max = (
                        torch.tensor(norm_stats["nlayers_data_max"])
                        .float()
                        .reshape(1, -1)
                    )
                    module.target_min = norm_stats["nlayers_target_min"]
                    module.target_max = norm_stats["nlayers_target_max"]
                    break
        else:
            # Fallback: use known data range for n_layers (25-45)
            print("  Warning: Using default n_layers range [25, 45] for MinMaxScaler")
            for module in nlayers_transform.sub_modules:
                if "MinMax" in module.__class__.__name__:
                    module.data_min = torch.tensor([[25.0]])
                    module.data_max = torch.tensor([[45.0]])
                    break

    return transforms["num_points"], transforms["energy"], nlayers_transform


def heun_sampler(model, x0, condition, n_steps=200, temperature=1.0):
    """Heun's method ODE solver for flow matching.

    temperature: scales the initial noise x0 by this factor (default 1.0 = no-op).
    Lower values compress output variance → reduces tail outliers.
    """
    x, dt = x0 * temperature, 1.0 / n_steps

    with torch.no_grad():
        for step in range(n_steps):
            t = step * dt
            t_next = min((step + 1) * dt, 1.0)

            t_tensor = torch.full((x.shape[0], 1), t, device=x.device)
            t_next_tensor = torch.full((x.shape[0], 1), t_next, device=x.device)

            v1 = model(x, t_tensor, condition)
            v2 = model(x + v1 * dt, t_next_tensor, condition)
            x = x + 0.5 * dt * (v1 + v2)

    return x


def apply_bias_correction(gen_counts, bias_correction_path, energy_bin=None):
    """Apply pre-computed bias correction: per-layer multiplicative + optional per-bin total.

    Args:
        gen_counts: int32 array, shape [N, D], raw model output after inverse_transform
        bias_correction_path: path to .npz file produced by compute_bias_correction.py
        energy_bin: optional (e_lo, e_hi) tuple — if provided and npz contains
                    total_bc_per_bin, applies the residual total-hits correction
                    for that energy bin after per-layer correction.

    Returns:
        Corrected int32 array of same shape.
    """
    bc = np.load(bias_correction_path)
    nl = int(bc["n_layers"])
    bf = bc["bias_factor"]  # shape [nl]
    corrected = np.round(gen_counts[:, :nl].astype(np.float64) * bf[np.newaxis, :])
    corrected = np.clip(corrected, 0, np.iinfo(np.int32).max).astype(np.int32)
    result = gen_counts.copy()
    result[:, :nl] = corrected

    # Per-bin total correction (residual scale after per-layer BC)
    if energy_bin is not None and "total_bc_per_bin" in bc:
        e_lo, e_hi = energy_bin
        bin_los = bc["energy_bin_lo"]
        bin_his = bc["energy_bin_hi"]
        for i in range(len(bin_los)):
            if (
                abs(float(bin_los[i]) - e_lo) < 1.0
                and abs(float(bin_his[i]) - e_hi) < 1.0
            ):
                total_factor = float(bc["total_bc_per_bin"][i])
                scaled = np.round(result[:, :nl].astype(np.float64) * total_factor)
                result[:, :nl] = np.clip(scaled, 0, np.iinfo(np.int32).max).astype(
                    np.int32
                )
                break

    return result


def inverse_transform(x, transform):
    """Apply inverse transform and convert to valid counts."""
    x_tensor = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
    x_inv = transform.inverse(x_tensor.float()).cpu().numpy()
    x_inv = np.nan_to_num(x_inv, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.round(x_inv), 0, np.iinfo(np.int32).max).astype(np.int32)
