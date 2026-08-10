"""Shared PCFM conditioning + checkpoint-loading helpers.

Single source of truth for the downstream (single-detector) conditioning
contract and PCFM checkpoint loading used by generate_pcfm_cond.py. Defining
build_conditions here means the condition column order
[norm_E, norm_SF, norm_NL, (dir_x, dir_y, dir_z)] and its normalization live
in exactly one place.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from model import build_model_from_config, remap_state_dict_for_dropout
from pcfm_utils import create_transforms


def load_pcfm_model(model_dir: Path, device, log_prefix: str = "    "):
    for ckpt_name in ("best_physics_model.pt", "best_model.pt"):
        ckpt_path = model_dir / ckpt_name
        if ckpt_path.exists():
            break
    else:
        raise FileNotFoundError(f"No checkpoint in {model_dir}")

    with open(model_dir / "conf.yaml") as f:
        config = yaml.safe_load(f)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    norm_stats = ckpt["norm_stats"]
    transform, _, _ = create_transforms(config, norm_stats)

    model = build_model_from_config(config, device)

    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        model.load_state_dict(
            remap_state_dict_for_dropout(ckpt["model_state_dict"]), strict=False
        )
    model.eval()
    print(
        f"{log_prefix}Loaded {ckpt_name} (epoch {ckpt.get('epoch', '?')}) from {model_dir.name}"
    )
    return model, transform, norm_stats, config


def build_conditions(energies_gev, dirs, norm_stats, nl_val, sf_val, device):
    n = len(energies_gev)
    e_mean = float(np.atleast_1d(norm_stats["energy_mean"]).item())
    e_std = float(np.atleast_1d(norm_stats["energy_std"]).item())
    norm_e = (np.log(energies_gev) - e_mean) / e_std

    f_min = float(np.atleast_1d(norm_stats["fsamp_data_min"]).item())
    f_max = float(np.atleast_1d(norm_stats["fsamp_data_max"]).item())
    f_tmin, f_tmax = norm_stats["fsamp_target_min"], norm_stats["fsamp_target_max"]
    norm_f = (
        (f_tmin + f_tmax) / 2.0
        if f_max == f_min
        else (sf_val - f_min) / (f_max - f_min) * (f_tmax - f_tmin) + f_tmin
    )

    nl_min = float(np.atleast_1d(norm_stats["nlayers_data_min"]).item())
    nl_max = float(np.atleast_1d(norm_stats["nlayers_data_max"]).item())
    nl_tmin, nl_tmax = (
        norm_stats["nlayers_target_min"],
        norm_stats["nlayers_target_max"],
    )
    norm_nl = (
        (nl_tmin + nl_tmax) / 2.0
        if nl_max == nl_min
        else (nl_val - nl_min) / (nl_max - nl_min) * (nl_tmax - nl_tmin) + nl_tmin
    )

    cols = [norm_e, np.full(n, norm_f), np.full(n, norm_nl)]
    if dirs is not None:
        cols.append(dirs.astype(np.float32))
    return torch.FloatTensor(np.column_stack(cols)).to(device)
