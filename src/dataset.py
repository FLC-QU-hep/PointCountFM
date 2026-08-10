"""
Multi-calorimeter dataset loader for flow matching training.
Handles conditioning on energy, sampling fraction, and optionally n_layers.
"""

import h5py
import torch

from preprocessing import Identity, Transformation, compose


class MultiCaloDataLoader:
    """DataLoader for multi-calorimeter shower data with conditioning."""

    def __init__(
        self,
        data_file: str,
        transform_energy: Transformation | list | None = None,
        transform_fsamp: Transformation | list | None = None,
        transform_num_points: Transformation | list | None = None,
        transform_nlayers: Transformation | list | None = None,
        batch_size: int = 1,
        shuffle: bool = False,
        start: int = 0,
        end: int | None = None,
        fit_transform: bool = False,
        device: torch.device | str = "cpu",
        use_nlayers_conditioning: bool = False,
        use_direction_conditioning: bool = False,
    ) -> None:
        import time

        t0 = time.time()

        self.data_file = data_file
        self.transform_energy = self._compose_trafo(transform_energy)
        self.transform_fsamp = self._compose_trafo(transform_fsamp)
        self.transform_num_points = self._compose_trafo(transform_num_points)
        self.transform_nlayers = self._compose_trafo(transform_nlayers)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_nlayers_conditioning = use_nlayers_conditioning
        self.use_direction_conditioning = use_direction_conditioning

        # Load data from HDF5
        print("  Loading data from HDF5...", end="", flush=True)
        t1 = time.time()
        with h5py.File(self.data_file, "r") as f:
            energy = f["energy"][start:end]
            sampling_fraction = f["sampling_fraction"][start:end]
            num_points = f["num_points"][start:end]
            if use_nlayers_conditioning:
                if "n_layers" not in f:
                    raise ValueError(
                        "use_nlayers_conditioning=True but 'n_layers' not in data file"
                    )
                n_layers = f["n_layers"][start:end]
            else:
                n_layers = None
            if use_direction_conditioning:
                if "directions" not in f:
                    raise ValueError(
                        "use_direction_conditioning=True but 'directions' not in data file"
                    )
                directions = f["directions"][start:end]
            else:
                directions = None
        print(f" done ({time.time() - t1:.1f}s)")

        self.num_samples = len(energy)

        # Convert to tensors and move to device
        print(f"  Converting to tensors and moving to {device}...", end="", flush=True)
        t1 = time.time()
        energy = torch.from_numpy(energy).to(torch.get_default_dtype()).to(device)
        sampling_fraction = (
            torch.from_numpy(sampling_fraction).to(torch.get_default_dtype()).to(device)
        )
        num_points = (
            torch.from_numpy(num_points).to(torch.get_default_dtype()).to(device)
        )
        if n_layers is not None:
            n_layers = (
                torch.from_numpy(n_layers).to(torch.get_default_dtype()).to(device)
            )
        if directions is not None:
            directions = (
                torch.from_numpy(directions).to(torch.get_default_dtype()).to(device)
            )
        print(f" done ({time.time() - t1:.1f}s)")

        # Apply transformations
        if fit_transform:
            print("  Fitting transforms...", end="", flush=True)
            t1 = time.time()
            energy = self.transform_energy.fit(energy)
            print(f" energy ({time.time() - t1:.1f}s)", end="", flush=True)

            t1 = time.time()
            sampling_fraction = self.transform_fsamp.fit(sampling_fraction)
            print(f", f_samp ({time.time() - t1:.1f}s)", end="", flush=True)

            t1 = time.time()
            num_points = self.transform_num_points.fit(num_points)
            print(f", num_points ({time.time() - t1:.1f}s)", end="", flush=True)

            if n_layers is not None:
                t1 = time.time()
                n_layers = self.transform_nlayers.fit(n_layers)
                print(f", n_layers ({time.time() - t1:.1f}s)", end="", flush=True)
            if directions is not None:
                print(", directions (unit vectors, no fit)", end="", flush=True)
            print()
        else:
            print("  Applying transforms...", end="", flush=True)
            t1 = time.time()
            energy = self.transform_energy(energy)
            sampling_fraction = self.transform_fsamp(sampling_fraction)
            num_points = self.transform_num_points(num_points)
            if n_layers is not None:
                n_layers = self.transform_nlayers(n_layers)
            print(f" done ({time.time() - t1:.1f}s)")

        # Combine into condition vector: [energy, fsamp, (n_layers), (dir_x, dir_y, dir_z)]
        parts = [energy, sampling_fraction]
        label = "energy, sampling_fraction"
        if n_layers is not None:
            parts.append(n_layers)
            label += ", n_layers"
        if directions is not None:
            parts.append(directions)
            label += ", direction(3)"
        self.condition = torch.cat(parts, dim=1)
        print(f"  Condition: [{label}] -> dim={self.condition.shape[1]}")

        self.data = num_points

        # PRE-GENERATE NOISE (like in paper)
        if fit_transform:  # Only for training
            print("  Pre-generating noise...", end="", flush=True)
            t1 = time.time()
            self.noise = torch.randn_like(self.data)
            print(f" done ({time.time() - t1:.1f}s)")
        else:  # Validation doesn't need noise
            self.noise = None

        print(f"  Total loader init time: {time.time() - t0:.1f}s")

    @staticmethod
    def _compose_trafo(transformation: Transformation | list | None) -> Transformation:
        """Compose transformation from list or return Identity."""
        if transformation is None:
            return Identity()
        if isinstance(transformation, list):
            return compose(transformation)
        return transformation

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def __iter__(self):
        indices = (
            torch.randperm(self.num_samples)
            if self.shuffle
            else torch.arange(self.num_samples)
        )
        for i in range(len(self)):
            idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
            batch = {
                "data": self.data[idx],
                "condition": self.condition[idx],
                "noise": self.noise[idx]
                if self.noise is not None
                else torch.randn_like(self.data[idx]),
            }
            yield batch

    def to(self, device_dtype: torch.device | torch.dtype | str) -> None:
        """Move all data and transforms to device/dtype."""
        self.data = self.data.to(device_dtype)
        self.condition = self.condition.to(device_dtype)
        if self.noise is not None:
            self.noise = self.noise.to(device_dtype)
        self.transform_energy.to(device_dtype)
        self.transform_fsamp.to(device_dtype)
        self.transform_num_points.to(device_dtype)
        self.transform_nlayers.to(device_dtype)
        # directions are unit vectors stored directly in self.condition — no separate tensor


def get_dataloaders(
    data_file: str,
    batch_size: int,
    batch_size_val: int,
    train_fraction: float = 0.9,
    val_samples: int | None = None,
    max_samples: int | None = None,
    transform_num_points: list | None = None,
    transform_inc: list | None = None,  # incident energy transform
    transform_fsamp: list | None = None,  # sampling fraction transform
    transform_nlayers: list | None = None,  # n_layers transform
    use_nlayers_conditioning: bool = False,  # Enable n_layers conditioning
    use_direction_conditioning: bool = False,  # Enable 3D direction conditioning
    device: torch.device | str = "cuda",
    pretrain_norm_stats: dict | None = None,
    data_offset: int = 0,
    **kwargs,
) -> tuple:
    """
    Create train and validation dataloaders.

    Parameters
    ----------
    data_file : str
        Path to HDF5 data file
    batch_size : int
        Training batch size
    batch_size_val : int
        Validation batch size
    train_fraction : float
        Fraction of data for training
    max_samples : int, optional
        Maximum samples to use
    transform_num_points : list, optional
        Transform config for num_points
    transform_inc : list, optional
        Transform config for incident energy
    transform_nlayers : list, optional
        Transform config for n_layers (if use_nlayers_conditioning=True)
    use_nlayers_conditioning : bool
        Whether to condition on n_layers (requires 'n_layers' in data file)
    device : torch.device or str
        Device to use
    """
    # Determine dataset size
    with h5py.File(data_file, "r") as f:
        total_samples = len(f["energy"])
        has_nlayers = "n_layers" in f
        if has_nlayers:
            max_layers = f["num_points"].shape[1]

    if use_nlayers_conditioning and not has_nlayers:
        raise ValueError(
            "use_nlayers_conditioning=True but 'n_layers' not found in data file. "
            "Use extract_training_data_variable_layers() to create the data."
        )

    if val_samples is not None:
        # val_samples is ADDITIVE to max_samples (training samples).
        # Training: samples [0 : n_train], Validation: samples [n_train : n_train + n_val]
        n_train = (
            min(max_samples, total_samples - val_samples)
            if max_samples is not None
            else max(0, total_samples - val_samples)
        )
        n_val = min(val_samples, total_samples - n_train)
    else:
        if max_samples is not None:
            total_samples = min(total_samples, max_samples)
        n_train = int(total_samples * train_fraction)
        n_val = total_samples - n_train

    # Multi-seed disjoint-block support: shift the TRAIN window by data_offset so each
    # seed trains on a non-overlapping consecutive block [data_offset : data_offset+n_train]
    # (e.g. D=1e3 seeds use [0:1000], [1000:2000], ...). Validation is then pinned to a FIXED
    # tail block (same for every seed, disjoint from all training blocks) so runs are comparable.
    # data_offset=0 (default) reproduces the original [0:n_train] / [n_train:n_train+n_val] split.
    train_start = data_offset
    train_end = data_offset + n_train
    if data_offset > 0:
        val_start = total_samples - n_val
        val_end = total_samples
        if train_end > val_start:
            raise ValueError(
                f"data_offset={data_offset}: train block [{train_start}:{train_end}] "
                f"overlaps the fixed val tail [{val_start}:{val_end}] (not enough events)"
            )
        print(
            f"  [multi-seed] train block [{train_start}:{train_end}], "
            f"fixed val tail [{val_start}:{val_end}]"
        )
    else:
        val_start, val_end = n_train, n_train + n_val

    # Ensure batch sizes never exceed the split size (avoids ZeroDivisionError)
    batch_size = min(batch_size, n_train)
    batch_size_val = min(batch_size_val, n_val)

    print(f"\nDataset: {n_train + n_val} samples ({n_train} train, {n_val} val)")
    if use_nlayers_conditioning:
        print(f"n_layers conditioning: ENABLED (max_layers={max_layers})")
    if use_direction_conditioning:
        print("direction conditioning: ENABLED (3D unit vector)")

    # Define transforms (use config if provided, otherwise defaults)
    if transform_inc is None:
        transform_energy = [
            ["Log", {"alpha": 1e-8}],
            ["StandardScaler", {"shape": [1, 1]}],
        ]
    else:
        transform_energy = transform_inc

    if transform_fsamp is None:
        transform_fsamp = [
            ["MinMaxScaler", {"shape": [1, 1], "target_min": -1.0, "target_max": 1.0}]
        ]

    # Use transform from config if provided, otherwise default
    if transform_num_points is None:
        dim = max_layers if use_nlayers_conditioning else 30
        transform_num_points = [
            ["Log", {"alpha": 1.0}],
            ["StandardScaler", {"shape": [1, dim]}],
        ]

    # n_layers transform (simple standardization)
    if transform_nlayers is None:
        transform_nlayers = [["StandardScaler", {"shape": [1, 1]}]]

    print(f"Transform energy: {[t[0] for t in transform_energy]}")
    print(f"Transform fsamp: {[t[0] for t in transform_fsamp]}")
    print(f"Transform num_points: {[t[0] for t in transform_num_points]}")
    if use_nlayers_conditioning:
        print(f"Transform n_layers: {[t[0] for t in transform_nlayers]}")

    # When pretrain_norm_stats is provided, build pre-fitted transforms
    # so the finetune model sees the same normalized inputs as the pretrain.
    if pretrain_norm_stats is not None:
        # Set stats on composed transforms so the finetune model sees the
        # same normalized inputs as the pretrain.
        from preprocessing import compose

        pt_energy = compose(transform_energy)
        pt_fsamp = compose(transform_fsamp)
        pt_numpts = compose(transform_num_points)
        pt_nlayers = compose(transform_nlayers) if use_nlayers_conditioning else None

        ns = pretrain_norm_stats
        for mod in pt_energy.sub_modules:
            if "Standard" in mod.__class__.__name__:
                mod.mean = torch.tensor(ns["energy_mean"]).float().reshape(1, -1)
                mod.std = torch.tensor(ns["energy_std"]).float().reshape(1, -1)
        for mod in pt_numpts.sub_modules:
            if "Standard" in mod.__class__.__name__:
                pt_mean = torch.tensor(ns["numpts_mean"]).float().reshape(1, -1)
                pt_std = torch.tensor(ns["numpts_std"]).float().reshape(1, -1)
                target_dim = transform_num_points[-1][1].get(
                    "shape", [1, pt_mean.shape[-1]]
                )[-1]
                if pt_mean.shape[-1] != target_dim:
                    # Resize: pad with mean-of-means / mean-of-stds for new layers
                    new_mean = pt_mean.mean().expand(1, target_dim).clone()
                    new_mean[:, : pt_mean.shape[-1]] = pt_mean
                    new_std = pt_std.mean().expand(1, target_dim).clone()
                    new_std[:, : pt_std.shape[-1]] = pt_std
                    print(
                        f"  Resized numpts norm_stats: {pt_mean.shape[-1]} → {target_dim}"
                    )
                    pt_mean, pt_std = new_mean, new_std
                mod.mean = pt_mean
                mod.std = pt_std
        for mod in pt_fsamp.sub_modules:
            if "MinMax" in mod.__class__.__name__:
                mod.data_min = torch.tensor(ns["fsamp_data_min"]).float().reshape(1, -1)
                mod.data_max = torch.tensor(ns["fsamp_data_max"]).float().reshape(1, -1)
            elif "Standard" in mod.__class__.__name__ and "fsamp_mean" in ns:
                mod.mean = torch.tensor(ns["fsamp_mean"]).float().reshape(1, -1)
                mod.std = torch.tensor(ns["fsamp_std"]).float().reshape(1, -1)
        if pt_nlayers is not None and "nlayers_data_min" in ns:
            for mod in pt_nlayers.sub_modules:
                if "MinMax" in mod.__class__.__name__:
                    mod.data_min = (
                        torch.tensor(ns["nlayers_data_min"]).float().reshape(1, -1)
                    )
                    mod.data_max = (
                        torch.tensor(ns["nlayers_data_max"]).float().reshape(1, -1)
                    )

        # Move transforms to the target device
        for t in [pt_energy, pt_fsamp, pt_numpts] + (
            [pt_nlayers] if pt_nlayers else []
        ):
            t.to(device)
        print("  Using pretrain transforms (overriding local fit)")

        # Both loaders use pre-fitted transforms, no fitting
        print("\n[TRAIN LOADER]")
        train_loader = MultiCaloDataLoader(
            data_file,
            transform_energy=pt_energy,
            transform_fsamp=pt_fsamp,
            transform_num_points=pt_numpts,
            transform_nlayers=pt_nlayers,
            batch_size=batch_size,
            shuffle=True,
            start=train_start,
            end=train_end,
            fit_transform=False,
            device=device,
            use_nlayers_conditioning=use_nlayers_conditioning,
            use_direction_conditioning=use_direction_conditioning,
        )
        print("\n[VAL LOADER]")
        val_loader = MultiCaloDataLoader(
            data_file,
            transform_energy=pt_energy,
            transform_fsamp=pt_fsamp,
            transform_num_points=pt_numpts,
            transform_nlayers=pt_nlayers,
            batch_size=batch_size_val,
            shuffle=False,
            start=val_start,
            end=val_end,
            fit_transform=False,
            device=device,
            use_nlayers_conditioning=use_nlayers_conditioning,
            use_direction_conditioning=use_direction_conditioning,
        )
        print()
        return train_loader, val_loader

    # Create training loader
    print("\n[TRAIN LOADER]")
    train_loader = MultiCaloDataLoader(
        data_file,
        transform_energy=transform_energy,
        transform_fsamp=transform_fsamp,
        transform_num_points=transform_num_points,
        transform_nlayers=transform_nlayers,
        batch_size=batch_size,
        shuffle=True,
        start=train_start,
        end=train_end,
        fit_transform=True,
        device=device,
        use_nlayers_conditioning=use_nlayers_conditioning,
        use_direction_conditioning=use_direction_conditioning,
    )

    # Create validation loader (reuse ALL transforms from training)
    print("\n[VAL LOADER]")
    val_loader = MultiCaloDataLoader(
        data_file,
        transform_energy=train_loader.transform_energy,
        transform_fsamp=train_loader.transform_fsamp,
        transform_num_points=train_loader.transform_num_points,
        transform_nlayers=train_loader.transform_nlayers,
        batch_size=batch_size_val,
        shuffle=False,
        start=val_start,
        end=val_end,
        fit_transform=False,
        device=device,
        use_nlayers_conditioning=use_nlayers_conditioning,
        use_direction_conditioning=use_direction_conditioning,
    )

    print()

    return train_loader, val_loader
