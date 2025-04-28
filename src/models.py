import torch
from torch import nn

__all__ = [
    "FullyConnected",
    "ConcatSquash",
]


class FullyConnected(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = dim_input + dim_condition + dim_time
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, dim_input))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        input = [t, x]
        if condition is not None:
            input.append(condition)
        input = torch.cat(input, dim=-1)
        return self.network(input)


class ConcatSquash(nn.Module):
    class __ConcatSquashLayer(nn.Module):
        def __init__(
            self,
            dim_input: int,
            dim_output: int,
            dim_cond: int,
            activation: type[nn.Module],
        ) -> None:
            super().__init__()
            self.cond_embed = nn.Sequential(
                nn.Linear(dim_cond, dim_input),
            )
            self.network = nn.Sequential(
                nn.Linear(dim_input, dim_output),
                activation(),
            )

        def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            cond_embed = self.cond_embed(condition)
            return self.network(x + cond_embed)

    def __init__(
        self,
        dim_input: int,
        dim_condition: int = 0,
        dim_time: int = 0,
        hidden_dims: list[int] | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev_dim = dim_input
        dim_cond_time = dim_condition + dim_time
        for dim in hidden_dims:
            layers.append(
                self.__ConcatSquashLayer(prev_dim, dim, dim_cond_time, activation)
            )
            prev_dim = dim
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(prev_dim, dim_input)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        condition = torch.cat([t, condition], dim=-1)
        for layer in self.layers:
            x = layer(x, condition)
        return self.output(x)
