import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from pointcountfm.preprocessing import Identity, Transformation, compose


def load_dataset(
    file: h5py.File,
    key: str,
    start: int = 0,
    end: int | None = None,
) -> Tensor:
    if key not in file:
        raise KeyError(f"Key '{key}' not found in HDF5 file.")
    dataset = file[key]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"Key '{key}' is not a dataset in HDF5 file.")
    data = dataset[start:end]
    is_integer = issubclass(data.dtype.type, (np.integer, np.bool_))
    tensor_dtype = torch.int64 if is_integer else torch.get_default_dtype()
    return torch.from_numpy(data).to(tensor_dtype, copy=False)


def load_data_file(
    data_file: str,
    start: int = 0,
    end: int | None = None,
    load_noise: bool = False,
    num_classes: int = 0,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    with h5py.File(data_file, "r") as file:
        energy_key = "energy" if "energy" in file else "energies"
        e_incident = load_dataset(file, energy_key, start, end)
        num_points = load_dataset(file, "num_points", start, end)
        noise = load_dataset(file, "noise", start, end) if load_noise else None
        labels = load_dataset(file, "labels", start, end) if "labels" in file else None
        directions = (
            load_dataset(file, "directions", start, end)
            if "directions" in file
            else None
        )

    if labels is not None:
        if not num_classes:
            num_classes = int(torch.max(labels).item()) + 1
        labels = F.one_hot(labels, num_classes=num_classes).to(
            torch.get_default_dtype()
        )

    return (
        e_incident,
        num_points,
        noise,
        labels,
        directions,
    )


def load_data_directory(
    data_directory: str,
    start: int = 0,
    end: int | None = None,
    load_noise: bool = False,
    num_classes: int = 0,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    e_incident_list = []
    num_points_list = []
    noise_list = []
    labels_list = []
    directions_list = []

    for i, file_name in enumerate(sorted(os.listdir(data_directory))):
        if not file_name.endswith(".h5"):
            continue
        file_path = os.path.join(data_directory, file_name)
        e_incident, num_points, noise, labels, directions = load_data_file(
            file_path, start=start, end=end, load_noise=load_noise
        )
        if labels is not None and not torch.all(torch.argmax(labels, dim=1) == i):
            raise ValueError(
                f"Labels in file {file_name} do not match expected label {i}."
            )
        e_incident_list.append(e_incident)
        num_points_list.append(num_points)
        noise_list.append(noise)
        labels_list.append(torch.full((e_incident.shape[0],), i, dtype=torch.int64))
        if directions is not None:
            directions_list.append(directions)

    e_incident = torch.cat(e_incident_list, dim=0)
    num_points = torch.cat(num_points_list, dim=0)
    noise = torch.cat(noise_list, dim=0) if load_noise else None
    labels = torch.cat(labels_list, dim=0)
    directions = torch.cat(directions_list, dim=0) if directions_list else None

    permutation = torch.randperm(e_incident.shape[0])
    e_incident = e_incident[permutation]
    num_points = num_points[permutation]
    if noise is not None:
        noise = noise[permutation]
    labels = labels[permutation]
    if not num_classes:
        num_classes = int(torch.max(labels).item()) + 1
    labels = F.one_hot(labels, num_classes=num_classes).to(torch.get_default_dtype())
    if directions is not None:
        directions = directions[permutation]

    return (
        e_incident,
        num_points,
        noise,
        labels,
        directions,
    )


def load_data(
    data_file: str,
    start: int = 0,
    end: int | None = None,
    load_noise: bool = False,
    num_classes: int = 0,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    if os.path.isdir(data_file):
        return load_data_directory(
            data_file, start, end, load_noise, num_classes=num_classes
        )
    elif os.path.isfile(data_file):
        return load_data_file(
            data_file, start, end, load_noise, num_classes=num_classes
        )
    else:
        raise ValueError(
            f"Data file {data_file} does not exist or is not a file or directory."
        )


class DataLoader:
    def __init__(
        self,
        data_file: str,
        transform_inc: Transformation | list | None = None,
        transform_num_points: Transformation | list | None = None,
        batch_size: int = 1,
        shuffle: bool = False,
        start: int = 0,
        end: int | None = None,
        fit_transform: bool = False,
        device: torch.device | str = "cpu",
        load_noise: bool = False,
        num_classes: int = 0,
    ) -> None:
        self.data_file = data_file
        self.transform_inc = self.__compose_trafo(transform_inc)
        self.transform_num_points = self.__compose_trafo(transform_num_points)
        self.batch_size = batch_size
        self.shuffle = shuffle

        e_incident, num_points, noise, labels, directions = load_data(
            data_file,
            start=start,
            end=end,
            load_noise=load_noise,
            num_classes=num_classes,
        )
        self.num_samples = len(e_incident)
        if labels is not None:
            self.num_classes = labels.shape[1]
        else:
            self.num_classes = 0
        self.energy_min = e_incident.min().item()
        self.energy_max = e_incident.max().item()

        e_incident = e_incident.to(device)
        num_points = num_points.to(device)
        if noise is not None:
            noise = noise.to(device)
        if labels is not None:
            labels = labels.to(device)
        if directions is not None:
            directions = directions.to(device)

        if fit_transform:
            e_incident = self.transform_inc.fit(e_incident)
            num_points = self.transform_num_points.fit(num_points)
        else:
            e_incident = self.transform_inc(e_incident)
            num_points = self.transform_num_points(num_points)

        self.data = num_points
        if labels is not None and directions is not None:
            self.condition = torch.cat([e_incident, labels, directions], dim=-1)
        elif labels is not None:
            self.condition = torch.cat([e_incident, labels], dim=-1)
        elif directions is not None:
            self.condition = torch.cat([e_incident, directions], dim=-1)
        else:
            self.condition = e_incident
        self.has_directions = directions is not None
        self.noise = noise

    @staticmethod
    def __compose_trafo(transformation: Transformation | list | None) -> Transformation:
        if transformation is None:
            return Identity()
        if isinstance(transformation, list):
            return compose(transformation)
        return transformation

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples)
        else:
            indices = torch.arange(self.num_samples)
        for i in range(len(self)):
            idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
            batch = {"data": self.data[idx], "condition": self.condition[idx]}
            if self.noise is not None:
                batch["noise"] = self.noise[idx]
            else:
                batch["noise"] = None
            yield batch

    def to(self, device_dtype: torch.device | torch.dtype | str) -> None:
        self.data = self.data.to(device_dtype)
        self.condition = self.condition.to(device_dtype)
        self.transform_inc.to(device_dtype)
        self.transform_num_points.to(device_dtype)
        if self.noise is not None:
            self.noise = self.noise.to(device_dtype)


def get_loaders(
    data_file: str,
    transform_inc: Transformation | list | None = None,
    transform_num_points: Transformation | list | None = None,
    batch_size: int = 128,
    batch_size_val: int | None = None,
    device: torch.device | str = "cpu",
    num_train: int | None = None,
    num_val: int = 10_000,
    load_noise: bool = False,
    num_classes: int = 0,
) -> tuple[DataLoader, DataLoader]:
    if num_train is None:
        num_train = -num_val
    if batch_size_val is None:
        batch_size_val = batch_size
    train_loader = DataLoader(
        data_file,
        transform_inc=transform_inc,
        transform_num_points=transform_num_points,
        batch_size=batch_size,
        shuffle=True,
        start=0,
        end=num_train,
        fit_transform=True,
        device=device,
        load_noise=load_noise,
        num_classes=num_classes,
    )
    test_loader = DataLoader(
        data_file,
        transform_inc=train_loader.transform_inc,
        transform_num_points=train_loader.transform_num_points,
        batch_size=batch_size_val,
        shuffle=False,
        start=-num_val,
        end=None,
        fit_transform=False,
        device=device,
        load_noise=load_noise,
        num_classes=num_classes,
    )
    return train_loader, test_loader
