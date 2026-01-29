import math

import torch
from torch import nn

__all__ = [
    "Transformation",
    "Sequence",
    "Identity",
    "Log",
    "LogIt",
    "Affine",
    "Clamp",
    "StandardScaler",
    "Dequantize",
    "compose",
]


class Transformation(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Sequence(Transformation):
    def __init__(self, modules: list[Transformation]) -> None:
        super().__init__()
        self.sub_modules = nn.ModuleList(modules)

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for module in self.sub_modules:
            if isinstance(module, Transformation):
                x = module.fit(x, mask)
            else:
                raise ValueError("All sub-modules must be of type Transformation")
        return x

    def forward(self, x: torch.Tensor):
        for module in self.sub_modules:
            x = module.forward(x)
        return x

    def inverse(self, x: torch.Tensor):
        for module in self.sub_modules[::-1]:
            if isinstance(module, Transformation):
                x = module.inverse(x)
            else:
                raise ValueError("All sub-modules must be of type Transformation")
        return x


class Identity(Transformation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Log(Transformation):
    def __init__(self, alpha: float = 1e-6, base: float = math.e) -> None:
        super().__init__()
        self.alpha = alpha
        self.log_base = math.log(base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.alpha) / self.log_base

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_base * x) - self.alpha


class LogIt(Transformation):
    def __init__(self, alpha: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (1 - 2 * self.alpha) * x + self.alpha
        return torch.log(x / (1 - x))

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(x)
        return (x - self.alpha) / (1 - 2 * self.alpha)


class Affine(Transformation):
    def __init__(self, scale: float = 1.0, shift: float = 0.0) -> None:
        super().__init__()
        self.a = scale
        self.b = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * x + self.b

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b) / self.a


class Clamp(Transformation):
    def __init__(self, min: float = 0.0, max: float = 1.0) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min, self.max)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x


class StandardScaler(Transformation):
    def __init__(self, shape: tuple[int]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))
        self.shape = shape

    def fit(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        dims = tuple(torch.where(torch.tensor(self.shape) == 1)[0].tolist())
        mean = torch.sum(x * mask, dim=dims, keepdim=True)
        mean /= torch.sum(mask, dim=dims, keepdim=True)
        self.mean = mean
        x = x - mean
        std = torch.sqrt(torch.sum(x**2 * mask, dim=dims, keepdim=True))
        std /= torch.sqrt(torch.sum(mask, dim=dims, keepdim=True) - 1)
        std[std == 0] = 1
        self.std = std
        x = x / std
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class Dequantize(Transformation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.rand_like(x)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.floor(x)


def compose(transformation: list[list[str | dict | list | None]] | None) -> Sequence:
    if transformation is None:
        return Sequence([Identity()])
    trafo_list = []
    attrs = globals()
    for element in transformation:
        if (element[0] not in __all__) or (
            element[0] in ["Transformation", "Sequence", "compose"]
        ):
            raise ValueError(f"Invalid transformation: {element[0]}")
        Trafo = attrs[element[0]]
        assert issubclass(Trafo, Transformation)
        if len(element) == 1 or element[1] is None:
            trafo_list.append(Trafo())
        elif isinstance(element[1], list):
            trafo_list.append(Trafo(*element[1]))
        elif isinstance(element[1], dict):
            trafo_list.append(Trafo(**element[1]))
        else:
            raise ValueError(
                f"argument for {element[0]} must be a list or a dict not {type(element[1])}"
            )

    return Sequence(trafo_list)
