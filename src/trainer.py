"""
Trainer for flow matching models.
Based on the original PointCountFM implementation.
"""

try:
    from comet_ml import Experiment

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

import argparse
import copy
import datetime
import os
import shutil
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy.optimize import linear_sum_assignment

from dataset import get_dataloaders
from model import remap_state_dict_for_dropout
from pcfm_utils import (
    create_norm_stats_dict,
    extract_minmax_stats,
    extract_scaler_scalar,
    heun_sampler,
    inverse_transform,
)
from preprocessing import detect_active_dims


def load_comet_config(
    config_path: str = "config/comet_config.yaml",
) -> dict | None:
    """Load Comet ML configuration from file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, Exception):
        return None


def _resize_state_dict(state_dict: dict, model: nn.Module) -> dict:
    """Resize pretrained weights to match the current model dimensions.

    When dim_input changes (e.g. 45 → 48), the first and last linear layers
    in the network have different shapes.  This function copies the overlapping
    weights and leaves new rows/columns at their freshly-initialized values.

    All other parameters are copied as-is.  Keys present in the state_dict
    but absent in the model (or vice versa) are silently skipped.
    """
    model_state = model.state_dict()
    resized = {}
    for key, pretrained_val in state_dict.items():
        if key not in model_state:
            continue
        target_shape = model_state[key].shape
        if pretrained_val.shape == target_shape:
            resized[key] = pretrained_val
        elif pretrained_val.ndim in (1, 2) and len(target_shape) == pretrained_val.ndim:
            # Partial copy: keep the model's freshly-initialized tensor,
            # overwrite the overlapping region with pretrained values.
            new_val = model_state[key].clone()
            slices = tuple(
                slice(0, min(p, t)) for p, t in zip(pretrained_val.shape, target_shape)
            )
            new_val[slices] = pretrained_val[slices]
            resized[key] = new_val
            print(
                f"  Resized {key}: {list(pretrained_val.shape)} → {list(target_shape)}"
            )
        else:
            resized[key] = pretrained_val
    return resized


class Trainer:
    def __init__(
        self,
        model_config: dict,
        data_config: dict,
        training_config: dict,
        device: torch.device,
        result_dir: str,
        use_comet: bool = True,
    ) -> None:
        self.device = device
        self.result_dir = result_dir
        self.epochs = training_config["epochs"]
        self.save_every = training_config.get("save_every", 100)
        self.snapshot_every = training_config.get("snapshot_every", 0)
        self.test_every = training_config.get("test_every", 50)
        self.patience = training_config.get("patience", 0)  # 0 = disabled
        self.epochs_no_improve = 0
        self.physics_patience = training_config.get(
            "physics_patience", 0
        )  # 0 = disabled
        self.physics_no_improve = 0
        self._dead_dims = training_config.get("dead_dims", None)
        self.max_grad_norm = training_config.get("max_grad_norm", None)

        data_config["device"] = self.device
        # Load pretrain norm_stats for transform alignment (finetune or scratch)
        pretrain_weights = training_config.get("pretrain_weights", None)
        transforms_from = training_config.get(
            "pretrain_transforms_from", pretrain_weights
        )
        if transforms_from and training_config.get("use_pretrain_transforms", False):
            pt_ckpt = torch.load(
                transforms_from, map_location=device, weights_only=False
            )
            data_config["pretrain_norm_stats"] = pt_ckpt.get("norm_stats")
            print(f"Using pretrain transforms from {transforms_from}")
        self.train_loader, self.val_loader = get_dataloaders(**data_config)
        self.data_config = data_config
        self.dim_data = self.train_loader.data.shape[1]
        self.dim_condition = self.train_loader.condition.shape[1]

        self.model = self.__init_model(model_config)

        # Load pretrained weights before optimizer init (only if no checkpoint exists yet)
        pretrain_weights = training_config.get("pretrain_weights", None)
        if pretrain_weights and not os.path.exists(
            os.path.join(result_dir, "checkpoint.pt")
        ):
            ckpt = torch.load(pretrain_weights, map_location=device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt.get("model", {}))
            try:
                self.model.load_state_dict(state)
            except RuntimeError:
                # Shape mismatch (e.g. dim_input changed) or dropout remap needed.
                remapped = remap_state_dict_for_dropout(state)
                resized = _resize_state_dict(remapped, self.model)
                self.model.load_state_dict(resized, strict=False)
            print(f"Loaded pretrained weights from {pretrain_weights}")

        # Freeze early layers if configured
        freeze_layers = training_config.get("freeze_layers", 0)
        if freeze_layers > 0:
            # Freeze time_embed and condition_embed
            for p in self.model.time_embed.parameters():
                p.requires_grad_(False)
            for p in self.model.condition_embed.parameters():
                p.requires_grad_(False)
            # Freeze first N hidden blocks in network Sequential
            # Each block is 2 modules (Linear+SiLU) or 3 (Linear+SiLU+Dropout)
            step = 3 if training_config.get("dropout", 0) > 0 else 2
            n_freeze_modules = freeze_layers * step
            for i, module in enumerate(self.model.network):
                if i < n_freeze_modules:
                    for p in module.parameters():
                        p.requires_grad_(False)
            n_frozen = sum(
                p.numel() for p in self.model.parameters() if not p.requires_grad
            )
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(
                f"Frozen {n_frozen} params, training {n_train} (freeze_layers={freeze_layers})"
            )

        self.optimizer = self.__init_optimizer(training_config["optimizer"])
        self.scheduler = self.__init_scheduler(training_config.get("scheduler", {}))

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

        self.loss_log = self.__get_file_path("losses.csv")
        self.model_path = self.__get_file_path("model.pt")
        self.best_model_path = self.__get_file_path("best_model.pt")
        self.best_physics_path = self.__get_file_path("best_physics_model.pt")
        self.checkpoint_path = self.__get_file_path("checkpoint.pt")
        self.best_physics_score = float("inf")  # mean normalised Wasserstein per layer
        self.physics_log = []  # [(epoch, score), ...]

        # Plots directories
        self.plots_dir = self.__get_file_path("plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        self.plots_test_dir = os.path.join(self.plots_dir, "test")
        os.makedirs(self.plots_test_dir, exist_ok=True)

        # Comparison directory (for sibling from_scratch <-> finetune comparison)
        run_type = os.path.basename(result_dir)
        if run_type in ("from_scratch", "finetune"):
            size_dir = os.path.dirname(result_dir)
            sibling = "finetune" if run_type == "from_scratch" else "from_scratch"
            self.sibling_loss_log = os.path.join(size_dir, sibling, "losses.csv")
            self.comparison_dir = os.path.join(size_dir, "comparison")
            self.run_type = run_type
            os.makedirs(self.comparison_dir, exist_ok=True)
        else:
            self.sibling_loss_log = None
            self.comparison_dir = None
            self.run_type = None

        # Track LR per step
        self.lr_history = []
        self.step_history = []
        self.global_step = 0

        # Initialize Comet ML
        self.experiment = None
        if use_comet and COMET_AVAILABLE:
            comet_config = load_comet_config()
            if comet_config:
                try:
                    self.experiment = Experiment(
                        api_key=comet_config.get("api_key"),
                        workspace=comet_config.get("workspace"),
                        project_name=comet_config.get("project_name", "pointcountfm"),
                    )
                    self.experiment.log_parameters(
                        {**training_config, "model": model_config, "data": data_config}
                    )
                    self.experiment.set_name(os.path.basename(result_dir))
                    print("✓ Comet ML initialized")
                except Exception as e:
                    print(f"⚠️  Failed to initialize Comet ML: {e}")
                    self.experiment = None
        elif use_comet and not COMET_AVAILABLE:
            print("⚠️  Comet ML not installed. Install: pip install comet-ml")

        # Detect active (non-constant) dimensions from the fitted StandardScaler.
        # Dead dims have std clamped to 1.0 by our guard; their z-values are
        # always 0 and carry no information.  Masking them from the loss lets
        # the model focus all gradient on the meaningful dims.
        self.active_mask = self._detect_active_dims()
        if self.active_mask is not None:
            n_active = int(self.active_mask.sum().item())
            print(f"Active-dim loss mask: {n_active}/{self.dim_data} dims active")

        # Finetune-only fixes applied once at the start of a fresh finetune run.
        # (skipped on resume — checkpoint already exists by the time we get here
        #  if we loaded pretrained weights and a checkpoint was found)
        fresh_finetune = pretrain_weights and not os.path.exists(
            os.path.join(result_dir, "checkpoint.pt")
        )
        if fresh_finetune:
            # 1) Zero dead-dim output rows in the last linear layer.
            #    During ODE integration the dead-dim velocities accumulate and
            #    feed back into the network input, corrupting active-dim outputs
            #    near the dead-dim boundary (L08-L10 overshoot).  Zeroing these
            #    rows stops that feedback loop.  The loss already masks dead dims
            #    so these rows receive zero gradient and stay at zero.
            last_linear = self.model.network[-1]
            if training_config.get("reset_output_layer", False):
                # Full reinit: all rows resampled from kaiming uniform.
                # Breaks the "L10 is not the last layer" encoding in the
                # active output rows, not just the dead ones.
                torch.nn.init.kaiming_uniform_(last_linear.weight, a=0.01)
                torch.nn.init.zeros_(last_linear.bias)
                print(
                    f"  Reset full output layer (all {last_linear.out_features} rows)"
                )
            elif (
                training_config.get("zero_dead_output_weights", True)
                and self.active_mask is not None
            ):
                dead_mask = self.active_mask.squeeze(0) == 0
                with torch.no_grad():
                    last_linear.weight.data[dead_mask] = 0.0
                    last_linear.bias.data[dead_mask] = 0.0
                print(f"  Zeroed dead-dim output weights ({int(dead_mask.sum())} rows)")

            # 2) Freeze early network layers so only the last `unfreeze_last` linear
            #    layers adapt during finetuning.  This prevents the model from
            #    re-learning SimpleBox representations while the output mapping adjusts.
            n_unfreeze = training_config.get("unfreeze_last_layers", 0)
            if n_unfreeze > 0:
                linears = [
                    m for m in self.model.network if isinstance(m, torch.nn.Linear)
                ]
                to_freeze = linears[:-n_unfreeze]
                for layer in to_freeze:
                    for p in layer.parameters():
                        p.requires_grad_(False)
                frozen_count = sum(
                    p.numel() for lin in to_freeze for p in lin.parameters()
                )
                total = sum(p.numel() for p in self.model.parameters())
                print(
                    f"  Frozen {len(to_freeze)}/{len(linears)} linear layers "
                    f"({frozen_count / total * 100:.1f}% of params)"
                )

        # Per-dimension loss weighting. mode="inverse_variance" (default) keeps the
        # original inverse-conditional-variance scheme; mode="deep_tail" up-weights the
        # under-filled deep tail layers past the occupancy peak (see _compute_loss_weights).
        self.use_loss_weighting = training_config.get("loss_weighting", False)
        self.loss_weight_mode = training_config.get(
            "loss_weight_mode", "inverse_variance"
        )
        self.tail_weight_max = training_config.get("tail_weight_max", 3.0)
        if self.use_loss_weighting:
            self.loss_weights = self._compute_loss_weights()
            print(f"Per-dim loss weighting: ENABLED (mode={self.loss_weight_mode})")
        else:
            self.loss_weights = None

        # Mini-batch Optimal Transport coupling
        self.use_ot = training_config.get("ot_coupling", False)
        if self.use_ot:
            print("Mini-batch OT coupling: ENABLED")

        # Logit-normal timestep sampling (Esser et al., SD3).
        # Concentrates training signal at t≈0 and t≈1 where ODE
        # integration error is highest.
        self.logit_normal_sigma = training_config.get("logit_normal_sigma", 0.0)
        if self.logit_normal_sigma > 0:
            print(f"Logit-normal timestep sampling: σ={self.logit_normal_sigma}")

        if os.path.exists(self.checkpoint_path):
            self.__load_checkpoint()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainer initialized: {n_params:,} parameters")
        print(
            f"Device: {self.device} | Train: {self.train_loader.data.shape[0]:,} | Val: {self.val_loader.data.shape[0]:,}"
        )
        sys.stdout.flush()

    def _detect_active_dims(self) -> torch.Tensor | None:
        """Return a ``[1, D]`` float mask of active dims, or *None*.

        If ``dead_dims`` is specified in the training config, use that directly
        instead of relying on scaler statistics (which may come from a pretrain
        with different active layers).
        """
        if self._dead_dims is not None:
            mask = torch.ones(1, self.dim_data, device=self.device)
            for d in self._dead_dims:
                mask[0, d] = 0.0
            return mask
        return detect_active_dims(self.train_loader.transform_num_points, self.device)

    def _compute_loss_weights(self) -> torch.Tensor:
        """Compute per-dimension loss weights from training data.

        Uses **inverse conditional variance**: bin training events by energy,
        compute per-dim variance within each bin, average across bins.  Dims
        whose conditional variance is high (noisy, stochastic) get down-weighted
        so the optimizer balances attention across all layers.

        This uses only training-set statistics — no data leakage.
        """
        if self.loss_weight_mode == "deep_tail":
            return self._loss_weights_deep_tail()
        data = self.train_loader.data  # [N, D] in z-space
        energies = self.train_loader.condition[:, 0]  # normalized log-energy
        n_bins = 10
        bin_edges = torch.quantile(
            energies,
            torch.linspace(0, 1, n_bins + 1, device=data.device),
        )
        cond_var = torch.zeros(data.shape[1], device=data.device)
        n_valid = 0
        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (energies >= bin_edges[i]) & (energies < bin_edges[i + 1])
            else:
                mask = (energies >= bin_edges[i]) & (energies <= bin_edges[i + 1])
            if mask.sum() > 1:
                cond_var += data[mask].var(dim=0)
                n_valid += 1
        if n_valid > 0:
            cond_var /= n_valid

        # Inverse-sqrt-variance weights, clamped at 1.0 from below.
        # Early layers (low cond. variance) get boosted above 1.0.
        # Late layers (high cond. variance) stay at exactly 1.0 — they
        # keep the same gradient as uniform MSE, never starved.
        eps = 1e-6
        weights = 1.0 / torch.sqrt(cond_var + eps)
        if self.active_mask is not None:
            weights = weights * self.active_mask.squeeze(0)

        # Normalize so the minimum active weight is exactly 1.0
        active_weights = weights[weights > eps]
        if len(active_weights) > 0:
            weights = weights / active_weights.min()  # min→1.0, rest >1.0

        print(
            f"  Loss weights (active dims): "
            f"min={weights[weights > eps].min():.3f}, "
            f"max={weights[weights > eps].max():.3f}, "
            f"ratio={weights[weights > eps].max() / weights[weights > eps].min():.2f}"
        )
        return weights.unsqueeze(0)  # [1, D]

    def _loss_weights_deep_tail(self) -> torch.Tensor:
        """Per-dim weights that UP-weight the under-filled DEEP tail layers.

        Layers up to and including the occupancy-peak layer keep weight 1.0; layers
        deeper than the peak get a linear ramp from 1.0 (just past the peak) to
        ``tail_weight_max`` at the deepest active layer.  This protects the
        systematically under-predicted deep layers — which plain MSE and the
        inverse-variance scheme leave at the 1.0 floor — WITHOUT boosting the noisy,
        low-occupancy shower front (which inverse-occupancy / direct-variance schemes
        wrongly inflate).  Uses train-split occupancy only — no data leakage.
        """
        # Recover per-layer mean occupancy in count space (monotonic inverse of the
        # Log+StandardScaler transform), train split only.
        counts = self.train_loader.transform_num_points.inverse(self.train_loader.data)
        occ = counts.float().mean(dim=0).flatten()  # [D]
        weights = torch.ones(occ.shape[0], device=occ.device)
        if self.active_mask is not None:
            active = torch.where(self.active_mask.squeeze(0) > 0)[0]
        else:
            active = torch.where(occ > 0.5)[0]
        if len(active) > 0:
            peak = int(active[occ[active].argmax()].item())
            last = int(active.max().item())
            wmax = float(self.tail_weight_max)
            if last > peak:
                for d in active.tolist():
                    if d > peak:
                        frac = (d - peak) / (last - peak)
                        weights[d] = 1.0 + frac * (wmax - 1.0)
            print(
                f"  Loss weights (deep_tail): peak=L{peak}, last=L{last}, wmax={wmax:.1f}, "
                f"tail={[round(float(weights[d]), 2) for d in active.tolist() if d > peak]}"
            )
        if self.active_mask is not None:
            weights = weights * self.active_mask.squeeze(0)
        return weights.unsqueeze(0)  # [1, D]

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample timesteps for flow matching.

        If ``logit_normal_sigma > 0``, uses logit-normal sampling:
        ``t = sigmoid(N(0, σ²))``.  This concentrates samples near t=0
        and t=1 where ODE integration error is highest (Esser et al.,
        Stable Diffusion 3).  Otherwise falls back to uniform U(0,1).
        """
        if self.logit_normal_sigma > 0:
            u = torch.randn(batch_size, 1, device=self.device)
            t = torch.sigmoid(u * self.logit_normal_sigma)
        else:
            t = torch.rand(batch_size, 1, device=self.device)
        return t

    @staticmethod
    def _ot_permutation(noise: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Find the OT-optimal permutation of *data* to match *noise*.

        Solves the linear assignment problem (equivalent to OT with uniform
        marginals) using scipy, returning the permutation indices so that
        ``data[perm]`` is the OT-paired reordering.
        """
        with torch.no_grad():
            cost = torch.cdist(noise, data, p=2)  # [B, B]
        _, col_ind = linear_sum_assignment(cost.cpu().numpy())
        return torch.from_numpy(col_ind).to(data.device)

    def __init_model(self, config: dict) -> nn.Module:
        config = config.copy()
        if "name" not in config:
            raise ValueError("Model configuration missing.")
        model_name = config.pop("name")

        if model_name == "FullyConnected":
            from model import FullyConnected as ModelClass
        elif model_name == "ConcatSquash":
            from model import ConcatSquash as ModelClass
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model = ModelClass(**config)
        return model.to(self.device)

    def __init_optimizer(self, config: dict) -> optim.Optimizer:
        config = config.copy()
        optimizer_name = config.pop("name")
        optimizer_class = getattr(optim, optimizer_name)
        if "lr" in config:
            self.lr = config["lr"]
        else:
            raise ValueError("Learning rate missing.")
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optimizer_class(params, **config)

    def __init_scheduler(self, config: dict) -> optim.lr_scheduler._LRScheduler:
        config = config.copy()
        if "name" not in config:
            return None
        scheduler_name = config.pop("name")
        if scheduler_name == "CosineAnnealingWarmup":
            warmup_epochs = config.pop("warmup_epochs", 200)
            warmup_steps = warmup_epochs * len(self.train_loader)
            T_max = config.pop("T_max", self.epochs) * len(self.train_loader)
            return optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    optim.lr_scheduler.LinearLR(
                        self.optimizer, start_factor=0.01, total_iters=warmup_steps
                    ),
                    optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer, T_max=T_max - warmup_steps
                    ),
                ],
                milestones=[warmup_steps],
            )
        scheduler_class = getattr(optim.lr_scheduler, scheduler_name)
        if scheduler_class is optim.lr_scheduler.OneCycleLR:
            config["max_lr"] = self.lr
            total_steps = len(self.train_loader) * self.epochs
            # PyTorch bug: total_steps=2 with pct_start=0.5 causes ZeroDivisionError
            config["total_steps"] = max(total_steps, 3)
        elif scheduler_class is optim.lr_scheduler.CosineAnnealingLR:
            # If T_max is given in yaml (in epochs), convert to steps.
            # Otherwise decay over the full training run.
            if "T_max" in config:
                config["T_max"] = config["T_max"] * len(self.train_loader)
            else:
                config["T_max"] = len(self.train_loader) * self.epochs
        elif scheduler_class is optim.lr_scheduler.CosineAnnealingWarmRestarts:
            # T_0 is given in epochs; convert to steps (scheduler.step() is
            # called once per batch).
            config["T_0"] = config.get("T_0", 200) * len(self.train_loader)
            if "T_mult" not in config:
                config["T_mult"] = 1
        return scheduler_class(self.optimizer, **config)

    def __get_file_path(self, filename: str) -> str:
        full_path = os.path.join(self.result_dir, filename)
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return full_path

    def __save_checkpoint(self) -> None:
        """Save checkpoint with normalization stats."""
        nlayers_transform = getattr(self.train_loader, "transform_nlayers", None)
        norm_stats = create_norm_stats_dict(
            self.train_loader.transform_energy,
            self.train_loader.transform_fsamp,
            self.train_loader.transform_num_points,
            nlayers_transform,
        )

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_losses": torch.tensor(self.train_losses),
            "val_losses": torch.tensor(self.val_losses),
            "best_val_loss": self.best_val_loss,
            "best_physics_score": self.best_physics_score,
            "norm_stats": norm_stats,
            "lr_history": self.lr_history,
            "step_history": self.step_history,
            "global_step": self.global_step,
            "epochs_no_improve": self.epochs_no_improve,
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        torch.save(checkpoint, self.checkpoint_path)

    def __load_checkpoint(self) -> None:
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_losses = checkpoint["train_losses"].tolist()
        self.val_losses = checkpoint["val_losses"].tolist()
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.best_physics_score = checkpoint.get("best_physics_score", float("inf"))
        self.lr_history = checkpoint.get("lr_history", [])
        self.step_history = checkpoint.get("step_history", [])
        self.global_step = checkpoint.get("global_step", 0)
        self.epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def __save_losses(self) -> None:
        with open(self.loss_log, "w") as file:
            for epoch, (train_loss, val_loss) in enumerate(
                zip(self.train_losses, self.val_losses)
            ):
                file.write(f"{epoch + 1} {train_loss} {val_loss}\n")

    def __save_best_model(self) -> None:
        """Save the best model weights with norm stats."""
        nlayers_transform = getattr(self.train_loader, "transform_nlayers", None)
        norm_stats = create_norm_stats_dict(
            self.train_loader.transform_energy,
            self.train_loader.transform_fsamp,
            self.train_loader.transform_num_points,
            nlayers_transform,
        )

        torch.save(
            {
                "epoch": len(self.train_losses),
                "model_state_dict": self.model.state_dict(),
                "val_loss": self.best_val_loss,
                "norm_stats": norm_stats,
            },
            self.best_model_path,
        )

        if self.experiment:
            self.experiment.log_model("best_model", self.best_model_path)

    def __plot_losses(self) -> None:
        """Plot training and validation loss vs epoch as PDF."""
        fig, ax = plt.subplots(figsize=(10, 7))
        epochs = np.arange(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, label="Train", color="#1f77b4", lw=2)
        ax.plot(epochs, self.val_losses, label="Validation", color="#ff7f0e", lw=2)
        ax.set(xlabel="Epoch", ylabel="Loss", yscale="log")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "loss_vs_epoch.pdf"), bbox_inches="tight"
        )
        plt.close()

    def __plot_loss_comparison(self) -> None:
        """Scratch-vs-finetune loss comparison lives in the analysis stack; no-op here."""
        return

    def __plot_wasserstein_comparison(self) -> None:
        """Plot Wasserstein distance vs epoch for FT and SC in comparison dir."""
        if self.comparison_dir is None or not self.physics_log:
            return
        try:
            my_epochs = [e for e, _ in self.physics_log]
            my_scores = [s for _, s in self.physics_log]

            # Load sibling physics log if exists
            sibling_log = os.path.join(
                self.comparison_dir,
                "physics_log_from_scratch.csv"
                if self.run_type == "finetune"
                else "physics_log_finetune.csv",
            )
            # Save my own log
            my_log_name = f"physics_log_{self.run_type}.csv"
            np.savetxt(
                os.path.join(self.comparison_dir, my_log_name),
                np.column_stack([my_epochs, my_scores]),
                header="epoch wasserstein",
            )

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.plot(
                my_epochs,
                my_scores,
                "o-",
                lw=2,
                ms=3,
                color="#d62728" if self.run_type == "finetune" else "#1f77b4",
                label=self.run_type,
            )

            if os.path.exists(sibling_log):
                sib = np.loadtxt(sibling_log)
                if sib.ndim == 2 and sib.shape[1] >= 2:
                    ax.plot(
                        sib[:, 0],
                        sib[:, 1],
                        "o-",
                        lw=2,
                        ms=3,
                        color="#1f77b4" if self.run_type == "finetune" else "#d62728",
                        label="finetune" if self.run_type != "finetune" else "scratch",
                    )

            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-2, top=1e3)
            ax.set(xlabel="Epoch", ylabel="Mean Wasserstein distance (per layer)")
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = os.path.join(self.comparison_dir, "comparison_wasserstein.pdf")
            plt.savefig(out, bbox_inches="tight")
            plt.close()
        except Exception:
            pass

    def __plot_lr(self) -> None:
        """Plot learning rate vs step as PDF."""
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(self.step_history, self.lr_history, color="#2ca02c", lw=2)
        ax.set(xlabel="Step", ylabel="Learning Rate")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "lr_vs_step.pdf"), bbox_inches="tight")
        plt.close()

    def __build_conditions(
        self, energies, e_mean, e_std, fsamp_val, nl_val, direction=None
    ):
        """Build normalized condition tensor for generation."""
        n = len(energies)
        log_energies = (np.log(energies) - e_mean) / e_std

        fsamp_minmax = extract_minmax_stats(self.train_loader.transform_fsamp)
        if fsamp_minmax is not None:
            f_min, f_max = (
                fsamp_minmax["data_min"].item(),
                fsamp_minmax["data_max"].item(),
            )
            f_tgt_min, f_tgt_max = (
                fsamp_minmax["target_min"],
                fsamp_minmax["target_max"],
            )
            if f_max == f_min:
                # Constant fsamp (e.g. Allegro): map to target center
                norm_f = (f_tgt_min + f_tgt_max) / 2.0
            else:
                norm_f = (fsamp_val - f_min) / (f_max - f_min) * (
                    f_tgt_max - f_tgt_min
                ) + f_tgt_min
        else:
            f_mean, f_std = extract_scaler_scalar(self.train_loader.transform_fsamp)
            norm_f = (fsamp_val - f_mean) / f_std

        nlayers_transform = getattr(self.train_loader, "transform_nlayers", None)
        norm_nl = nlayers_transform(
            torch.tensor([[nl_val]], dtype=torch.float32)
        ).item()

        cols = [log_energies, np.full(n, norm_f), np.full(n, norm_nl)]

        if getattr(self.train_loader, "use_direction_conditioning", False):
            if direction is None:
                direction = np.array([0.0, 0.0, 1.0])
            direction = np.asarray(direction, dtype=np.float32)
            # support both single direction (3,) and per-sample directions (n, 3)
            if direction.ndim == 1:
                cols.append(np.tile(direction, (n, 1)))
            else:
                cols.append(direction)

        return torch.FloatTensor(np.column_stack(cols)).to(self.device)

    def test_and_plot(self, epoch: int, data_file: str) -> float | None:
        """Generate test plots (longitudinal profiles + total hits) as PDF.

        Automatically selects the plot style:
        - Variable n_layers / SF (e.g. SimpleBox): vary n_layers and SF
        - Fixed n_layers and SF (e.g. Allegro):    vary energy in 5 bins

        Uses the **current** training weights so the physics score tracks
        the actual training evolution.  Restores weights after plotting.

        Returns
        -------
        float or None
            Physics score = mean |Model/G4 − 1| across ratio plots.
            Lower is better.  None if test failed.
        """
        # Use current training weights for physics evaluation
        current_state = copy.deepcopy(self.model.state_dict())
        try:
            physics_score = None
            use_dir = getattr(self.train_loader, "use_direction_conditioning", False)

            # Use only held-out events (index >= max_samples) as reference to avoid
            # any overlap with training data.  If the file has no held-out events
            # (e.g. 100k model trained on all 100k), fall back to the full file.
            # Skip training + validation data for physics eval
            n_train = len(self.train_loader.data)
            n_val = len(self.val_loader.data)
            eval_offset = n_train + n_val

            with h5py.File(data_file, "r") as f:
                total_in_file = f["num_points"].shape[0]
                if eval_offset >= total_in_file:
                    print(
                        f"  [WARN] No held-out events (max_samples={eval_offset} >= "
                        f"file size {total_in_file}). Using full file for physics eval."
                    )
                    eval_offset = 0
                real_data = f["num_points"][eval_offset:]
                real_energy = f["energy"][eval_offset:].flatten()
                real_fsamp = f["sampling_fraction"][eval_offset:].flatten()
                real_nlayers = (
                    f["n_layers"][eval_offset:].flatten() if "n_layers" in f else None
                )
                real_directions = (
                    f["directions"][eval_offset:]
                    if (use_dir and "directions" in f)
                    else None
                )

            if real_nlayers is None:
                return

            epoch_dir = os.path.join(self.plots_test_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)

            n_samples = 5000
            n_steps = 200
            e_mean, e_std = extract_scaler_scalar(self.train_loader.transform_energy)
            results, real_results = {}, {}

            # Allegro-like: fixed SF (std/mean < 1%), n_layers may vary (5..11)
            # SimpleBox-like: variable SF across a wide range
            allegro_like = np.std(real_fsamp) / (np.mean(real_fsamp) + 1e-8) < 0.01

            if allegro_like:
                # ── Allegro: fix nl=max (e.g. 11), sf≈0.162, vary energy ──────
                self.model.eval()
                nl = int(np.max(real_nlayers))  # user-specified: n_layers = 11
                sf = float(np.median(real_fsamp))  # user-specified: sf = 0.162

                # Use only showers with this n_layers for real reference
                nl_mask = real_nlayers == nl
                real_energy_nl = real_energy[nl_mask]
                real_data_nl = real_data[nl_mask]
                real_dirs_nl = (
                    real_directions[nl_mask] if real_directions is not None else None
                )

                percentiles = np.percentile(real_energy_nl, np.linspace(0, 100, 6))
                energy_bins = [
                    (float(percentiles[i]), float(percentiles[i + 1])) for i in range(5)
                ]

                with torch.no_grad():
                    for idx, ebin in enumerate(energy_bins):
                        e_lo, e_hi = ebin
                        mask = (real_energy_nl >= e_lo) & (real_energy_nl <= e_hi)
                        pool = np.where(mask)[0]
                        if len(pool) == 0:
                            continue
                        np.random.seed(42 + idx)
                        chosen = np.random.choice(pool, size=n_samples, replace=True)
                        energies = real_energy_nl[chosen]
                        dirs = (
                            real_dirs_nl[chosen] if real_dirs_nl is not None else None
                        )

                        cond = self.__build_conditions(
                            energies, e_mean, e_std, sf, nl, direction=dirs
                        )
                        torch.manual_seed(42 + idx)
                        x0 = torch.randn(n_samples, self.dim_data, device=self.device)
                        if self.active_mask is not None:
                            x0 = x0 * self.active_mask  # zero dead dims
                        results[ebin] = inverse_transform(
                            heun_sampler(self.model, x0, cond, n_steps),
                            self.val_loader.transform_num_points,
                        )
                        real_results[ebin] = real_data_nl[chosen]

                # ── Per-layer histograms with ratio panels ────────────────
                import math

                from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

                all_gen = np.concatenate(
                    [results[k] for k in energy_bins if k in results]
                )
                all_real = np.concatenate(
                    [real_results[k] for k in energy_bins if k in real_results]
                )
                n_cols = min(5, nl)
                n_rows = math.ceil(nl / n_cols)
                fig_lh = plt.figure(figsize=(n_cols * 4, n_rows * 5))
                fig_lh.suptitle(
                    f"Per-layer histograms (epoch {epoch})",
                    fontsize=16,
                    fontweight="bold",
                )
                outer = GridSpec(
                    n_rows,
                    n_cols,
                    figure=fig_lh,
                    hspace=0.5,
                    wspace=0.38,
                    top=0.93,
                    bottom=0.04,
                )

                for li in range(nl):
                    row, col = divmod(li, n_cols)
                    inner = GridSpecFromSubplotSpec(
                        2,
                        1,
                        subplot_spec=outer[row, col],
                        height_ratios=[3, 1],
                        hspace=0.0,
                    )
                    ax = fig_lh.add_subplot(inner[0])
                    axr = fig_lh.add_subplot(inner[1], sharex=ax)

                    vmax = max(all_real[:, li].max(), all_gen[:, li].max(), 2)
                    bins = np.logspace(0, np.log10(max(vmax, 2)), 40)
                    bc = (bins[:-1] + bins[1:]) / 2

                    g4_h, _ = np.histogram(all_real[:, li], bins=bins)
                    gen_h, _ = np.histogram(all_gen[:, li], bins=bins)
                    ax.stairs(
                        g4_h,
                        bins,
                        color="black",
                        lw=2,
                        label="Geant4",
                        fill=True,
                        facecolor="gray",
                        alpha=0.2,
                    )
                    ax.stairs(gen_h, bins, color="tab:red", lw=2, ls="--", label="Gen")

                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(g4_h > 0, gen_h / g4_h.astype(float), 1.0)
                    axr.plot(bc, ratio, color="tab:red", lw=1.5, ls="--")
                    axr.axhline(1, color="gray", ls="--", lw=1)
                    axr.set_xscale("log")
                    axr.set_ylim(0.5, 1.5)
                    axr.set_ylabel("Gen/G4", fontsize=8)
                    axr.set_xlabel("# Points", fontsize=8)
                    axr.tick_params(labelsize=7)

                    ax.set_title(f"Layer {li + 1}", fontsize=12)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_ylim(2e0, max(g4_h.max(), gen_h.max()) * 2)
                    ax.tick_params(labelbottom=False, labelsize=7)
                    if li == 0:
                        ax.legend(fontsize=8)

                fig_lh.savefig(
                    os.path.join(epoch_dir, "layer_hists.pdf"), bbox_inches="tight"
                )
                plt.close(fig_lh)

                # ── Compute physics metric ────────────────────────────────
                # Lower = better (0 = perfect).

                # ── Physics metric: mean normalized Wasserstein per layer
                from scipy.stats import wasserstein_distance

                layer_wd = []
                for ebin in energy_bins:
                    if ebin not in results:
                        continue
                    for li in range(nl):
                        gen_layer = results[ebin][:, li]
                        real_layer = real_results[ebin][:, li]
                        std_real = np.std(real_layer, ddof=1)
                        if std_real > 0:
                            wd = wasserstein_distance(gen_layer, real_layer) / std_real
                            layer_wd.append(wd)

                physics_score = np.mean(layer_wd) if layer_wd else None
                if physics_score is not None:
                    print(
                        f"  Physics score (mean norm. Wasserstein): {physics_score:.4f}"
                    )

                # ── Total hits bias: mean |gen_total/g4_total - 1| per energy bin
                all_gen = np.concatenate(
                    [results[k] for k in energy_bins if k in results]
                )
                all_real = np.concatenate(
                    [real_results[k] for k in energy_bins if k in real_results]
                )
                gen_total = all_gen[:, :nl].sum(axis=1).mean()
                real_total = all_real[:, :nl].sum(axis=1).mean()
                total_bias_pct = (gen_total / real_total - 1) * 100
                print(
                    f"  Total hits bias: gen={gen_total:.0f} vs G4={real_total:.0f} "
                    f"({total_bias_pct:+.1f}%)"
                )

            else:
                # ── SimpleBox-like: vary n_layers and SF ─────────────────────
                nlayers_list = [15, 22, 28, 36, 45]
                sf_list = [0.01, 0.02, 0.03, 0.04, 0.05]
                fixed_sf, fixed_nl = 0.03, 35

                def sample_real(target_sf, target_nl, seed):
                    nl_mask = real_nlayers == target_nl
                    sf_mask = np.abs(real_fsamp - target_sf) < 0.005
                    pool = np.where(nl_mask & sf_mask)[0]
                    if len(pool) == 0:
                        pool = np.where(nl_mask)[0]
                    np.random.seed(seed)
                    chosen = np.random.choice(pool, size=n_samples, replace=True)
                    dirs = (
                        real_directions[chosen] if real_directions is not None else None
                    )
                    return real_energy[chosen], real_data[chosen], dirs

                self.model.eval()
                with torch.no_grad():
                    for idx, nl in enumerate(nlayers_list):
                        key = (fixed_sf, nl)
                        energies, real_samples, dirs = sample_real(
                            fixed_sf, nl, 100 + idx
                        )
                        cond = self.__build_conditions(
                            energies, e_mean, e_std, fixed_sf, nl, direction=dirs
                        )
                        torch.manual_seed(100 + idx)
                        x0 = torch.randn(n_samples, self.dim_data, device=self.device)
                        if self.active_mask is not None:
                            x0 = x0 * self.active_mask  # zero dead dims
                        results[key] = inverse_transform(
                            heun_sampler(self.model, x0, cond, n_steps),
                            self.val_loader.transform_num_points,
                        )
                        real_results[key] = real_samples

                    for idx, sf in enumerate(sf_list):
                        key = (sf, fixed_nl)
                        if key in results:
                            continue
                        energies, real_samples, dirs = sample_real(
                            sf, fixed_nl, 200 + idx
                        )
                        cond = self.__build_conditions(
                            energies, e_mean, e_std, sf, fixed_nl, direction=dirs
                        )
                        torch.manual_seed(200 + idx)
                        x0 = torch.randn(n_samples, self.dim_data, device=self.device)
                        if self.active_mask is not None:
                            x0 = x0 * self.active_mask  # zero dead dims
                        results[key] = inverse_transform(
                            heun_sampler(self.model, x0, cond, n_steps),
                            self.val_loader.transform_num_points,
                        )
                        real_results[key] = real_samples

            plt.close("all")
            if allegro_like:
                print(f"  Test plots saved to {epoch_dir}")
            return physics_score

        except Exception as e:
            import traceback

            print(f"  Test plots failed: {e}")
            traceback.print_exc()
            return None
        finally:
            # Always restore current training weights after plotting
            self.model.load_state_dict(current_state)

    def train(self) -> None:
        print("Training started.")
        sys.stdout.flush()

        data_file = self.data_config.get(
            "data_file", "data/multicalo_energy_numpoints.h5"
        )

        for epoch in range(len(self.train_losses), self.epochs):
            self.model.train()
            train_loss = 0

            for batch in self.train_loader:
                self.optimizer.zero_grad()

                x = batch["data"]
                condition = batch["condition"]
                noise = batch["noise"]
                if self.active_mask is not None:
                    noise = (
                        noise * self.active_mask
                    )  # zero dead dims — no spurious noise as input

                # Mini-batch OT: permute data to match noise
                if self.use_ot:
                    perm = self._ot_permutation(noise, x)
                    x = x[perm]
                    condition = condition[perm]

                t = self._sample_timesteps(x.shape[0])
                x_t = t * x + (1 - t) * noise
                u_t = x - noise

                v_t = self.model(x_t, t, condition)
                sq_err = (v_t - u_t).square()
                if self.active_mask is not None:
                    sq_err = sq_err * self.active_mask  # zero out dead dims
                if self.loss_weights is not None:
                    sq_err = sq_err * self.loss_weights
                loss = sq_err.mean()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                train_loss += loss.item()
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1
                self.lr_history.append(self.optimizer.param_groups[0]["lr"])
                self.step_history.append(self.global_step)

            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in self.val_loader:
                    x = batch["data"]
                    condition = batch["condition"]
                    noise = torch.randn_like(x)
                    if self.active_mask is not None:
                        noise = noise * self.active_mask  # zero dead dims

                    t = self._sample_timesteps(x.shape[0])
                    x_t = t * x + (1 - t) * noise
                    u_t = x - noise

                    v_t = self.model(x_t, t, condition)
                    sq_err = (v_t - u_t).square()
                    if self.active_mask is not None:
                        sq_err = sq_err * self.active_mask
                    if self.loss_weights is not None:
                        sq_err = sq_err * self.loss_weights
                    loss = sq_err.mean()
                    val_loss += loss.item()

            val_loss /= len(self.val_loader)
            self.val_losses.append(val_loss)

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.__save_best_model()
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            # Save checkpoint regularly
            self.__save_checkpoint()
            if self.snapshot_every > 0 and (epoch + 1) % self.snapshot_every == 0:
                snap_dir = os.path.join(self.result_dir, "snapshots")
                os.makedirs(snap_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(snap_dir, f"epoch_{epoch + 1:05d}.pt"),
                )
            self.__save_losses()

            # Log to Comet
            if self.experiment:
                self.experiment.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "best_val_loss": self.best_val_loss,
                    },
                    step=epoch,
                )

            best_str = " ✓ NEW BEST" if is_best else ""
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{self.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}{best_str}"
            )
            sys.stdout.flush()

            # Update loss and LR plots every epoch
            self.__plot_losses()
            self.__plot_lr()
            self.__plot_loss_comparison()

            if (epoch + 1) % self.test_every == 0:
                physics_score = self.test_and_plot(epoch + 1, data_file)

                # Save epoch snapshot for retrospective analysis
                snapshot_path = os.path.join(
                    self.plots_test_dir,
                    f"epoch_{epoch + 1}",
                    "model_snapshot.pt",
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss": val_loss,
                        "physics_score": physics_score,
                    },
                    snapshot_path,
                )

                # Track best physics model (uses current training weights,
                # since test_and_plot evaluates the current snapshot)
                if physics_score is not None:
                    self.physics_log.append((epoch + 1, physics_score))
                    self.__plot_wasserstein_comparison()
                    if physics_score < self.best_physics_score:
                        self.best_physics_score = physics_score
                        # Save current training weights (what was evaluated)
                        nlayers_transform = getattr(
                            self.train_loader, "transform_nlayers", None
                        )
                        norm_stats = create_norm_stats_dict(
                            self.train_loader.transform_energy,
                            self.train_loader.transform_fsamp,
                            self.train_loader.transform_num_points,
                            nlayers_transform,
                        )
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "model_state_dict": self.model.state_dict(),
                                "val_loss": val_loss,
                                "physics_score": physics_score,
                                "norm_stats": norm_stats,
                            },
                            self.best_physics_path,
                        )
                        print(
                            f"  ✓ NEW BEST PHYSICS: {physics_score:.2f}% "
                            f"(epoch {epoch + 1})"
                        )
                        self.physics_no_improve = 0
                    else:
                        self.physics_no_improve += 1
                        print(
                            f"  Physics no-improve: {self.physics_no_improve}"
                            + (
                                f"/{self.physics_patience}"
                                if self.physics_patience > 0
                                else ""
                            )
                        )
                    sys.stdout.flush()

            # Early stopping — val loss
            if self.patience > 0 and self.epochs_no_improve >= self.patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: no improvement for "
                    f"{self.patience} epochs. Best val loss: {self.best_val_loss:.6f}"
                )
                sys.stdout.flush()
                break

            # Early stopping — physics score
            if (
                self.physics_patience > 0
                and self.physics_no_improve >= self.physics_patience
            ):
                print(
                    f"Physics early stopping at epoch {epoch + 1}: physics score "
                    f"no improvement for {self.physics_no_improve} evaluations. "
                    f"Best physics: {self.best_physics_score:.2f}%"
                )
                sys.stdout.flush()
                break

    def to(self, device_dtype: torch.device | torch.dtype | str) -> None:
        self.model.to(device_dtype)
        self.val_loader.to(device_dtype)
        self.train_loader.to(device_dtype)
        if isinstance(device_dtype, torch.device):
            self.device = device_dtype
        elif isinstance(device_dtype, str):
            device_dtype = device_dtype.lower()
            try:
                device = torch.device(device_dtype)
                self.device = device
            except RuntimeError:
                pass


def setup_result_path(run_name: str, conf_file: str, fast_dev_run: bool = False):
    """Create result directory and save config."""
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if fast_dev_run:
        result_path = os.path.join(repo_dir, "results/test")
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
    else:
        # Create unique timestamped directory
        now = datetime.datetime.now()
        while True:
            full_run_name = now.strftime("%Y%m%d_%H%M%S") + "_" + run_name
            result_path = os.path.join(repo_dir, "results", full_run_name)
            if not os.path.exists(result_path):
                break
            now += datetime.timedelta(seconds=1)

    os.makedirs(result_path, exist_ok=True)

    # Save config with result_path
    with open(conf_file) as f:
        lines = [line for line in f.readlines() if not line.startswith("result_path")]
    lines.insert(1, f"result_path: {result_path}\n")

    with open(os.path.join(result_path, "conf.yaml"), "w") as f:
        f.writelines(lines)

    return result_path


def parse_args(args: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    parser.add_argument(
        "--fast-dev-run", action="store_true", help="Run a test with a small dataset."
    )
    parser.add_argument(
        "-d",
        "--device",
        default="",
        type=str,
        help='Device to use for training (e.g., "cpu", "cuda", "mps").',
    )
    parser.add_argument(
        "--no-comet", action="store_true", help="Disable Comet ML tracking."
    )
    return parser.parse_args(args)


def main(args: list | None = None) -> None:
    args = parse_args(args)
    config = yaml.safe_load(open(args.config))

    # Optional global seed (model init + batch-shuffle order). Used for the D=1e5 band,
    # where the dataset is the full file so the only variance source is training stochasticity
    # (the disjoint-block scales vary the DATA instead). Default None = original (unseeded) behavior.
    seed = config.get("training", {}).get("seed")
    if seed is not None:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Global training seed set: {seed}")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.fast_dev_run:
        # Always use a clean results/test dir for fast-dev-run, regardless of config
        result_dir = setup_result_path(config["name"], args.config, fast_dev_run=True)
    elif "result_path" in config:
        result_dir = config["result_path"]
        os.makedirs(result_dir, exist_ok=True)
        conf_dst = os.path.join(result_dir, "conf.yaml")
        shutil.copy(args.config, conf_dst)
    else:
        result_dir = setup_result_path(config["name"], args.config, fast_dev_run=False)

    if args.fast_dev_run:
        config["data"]["max_samples"] = 1000
        config["training"]["epochs"] = 2
        config["training"]["test_every"] = 1

    trainer = Trainer(
        model_config=config["model"],
        data_config=config["data"],
        training_config=config["training"],
        device=device,
        result_dir=result_dir,
        use_comet=not args.no_comet,
    )
    trainer.train()


if __name__ == "__main__":
    main()
