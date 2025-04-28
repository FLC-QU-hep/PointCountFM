import h5py
import torch

from preprocessing import Identity, Transformation, compose


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
        ot_noise: bool = False,
        load_preprocessed: bool = False,
    ) -> None:
        self.data_file = data_file
        self.transform_inc = self.__compose_trafo(transform_inc)
        self.transform_num_points = self.__compose_trafo(transform_num_points)
        self.batch_size = batch_size
        self.shuffle = shuffle

        with h5py.File(self.data_file, "r") as f:
            e_incident = f["energy"][start:end]
            num_points = f["num_points"][start:end]
            if load_preprocessed:
                e_incident_pro = f["preprocessed_energy"][start:end]
                num_points_pro = f["preprocessed_data"][start:end]
            if ot_noise:
                noise = f["noise"][start:end]
            else:
                noise = None

        self.num_samples = len(e_incident)
        e_incident = torch.from_numpy(e_incident).to(
            torch.get_default_dtype(), copy=False
        )
        num_points = torch.from_numpy(num_points).to(
            torch.get_default_dtype(), copy=False
        )
        if noise is not None:
            noise = torch.from_numpy(noise).to(torch.get_default_dtype(), copy=False)
        if load_preprocessed:
            e_incident_pro = torch.from_numpy(e_incident_pro).to(
                torch.get_default_dtype(), copy=False
            )
            num_points_pro = torch.from_numpy(num_points_pro).to(
                torch.get_default_dtype(), copy=False
            )

        e_incident = e_incident.to(device)
        num_points = num_points.to(device)
        if noise is not None:
            noise = noise.to(device)
        if load_preprocessed:
            e_incident_pro = e_incident_pro.to(device)
            num_points_pro = num_points_pro.to(device)

        if fit_transform:
            e_incident = self.transform_inc.fit(e_incident)
            num_points = self.transform_num_points.fit(num_points)
        else:
            e_incident = self.transform_inc(e_incident)
            num_points = self.transform_num_points(num_points)

        if not load_preprocessed:
            self.data = num_points
            self.condition = e_incident
        else:
            self.data = num_points_pro
            self.condition = e_incident_pro
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
    batch_size_val: int = 0,
    device: torch.device | str = "cpu",
    num_train: int = 0,
    num_val: int = 0,
    ot_noise: bool = False,
    load_preprocessed: bool = False,
) -> tuple[DataLoader, DataLoader]:
    if bool(num_train) != bool(num_val):
        raise ValueError("Both num_train and num_val must be set or unset")
    train_loader = DataLoader(
        data_file,
        transform_inc=transform_inc,
        transform_num_points=transform_num_points,
        batch_size=batch_size,
        shuffle=True,
        start=0,
        end=-10_000 if num_train == 0 else num_train,
        fit_transform=True,
        device=device,
        ot_noise=ot_noise,
        load_preprocessed=load_preprocessed,
    )
    if batch_size_val == 0:
        batch_size_val = batch_size
    test_loader = DataLoader(
        data_file,
        transform_inc=train_loader.transform_inc,
        transform_num_points=train_loader.transform_num_points,
        batch_size=batch_size_val,
        shuffle=False,
        start=-10_000 if num_val == 0 else -num_val,
        end=None,
        device=device,
        ot_noise=ot_noise,
        load_preprocessed=load_preprocessed,
    )
    return train_loader, test_loader
