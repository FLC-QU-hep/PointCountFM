# model.py

import torch
import torch.nn as nn


def remap_state_dict_for_dropout(state_dict: dict) -> dict:
    """Remap a state dict trained *without* dropout to a model *with* dropout.

    Without dropout the ``network`` Sequential has layers at indices
    ``0, 2, 4, …`` (Linear, SiLU pairs).  With dropout the indices become
    ``0, 3, 6, …`` (Linear, SiLU, Dropout triples).  This function adjusts
    the keys accordingly.
    """
    remapped = {}
    for key, val in state_dict.items():
        if key.startswith("network."):
            parts = key.split(".")
            idx = int(parts[1])
            if idx % 2 == 0:
                parts[1] = str((idx // 2) * 3)
            remapped[".".join(parts)] = val
        else:
            remapped[key] = val
    return remapped


class FullyConnected(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_condition,
        dim_time,
        hidden_dims,
        dropout=0.0,
        adapter_dim=None,
        output_adapter_hidden=None,
    ):
        super().__init__()
        self.adapter_dim = adapter_dim
        # When adapter_dim is set, the core network operates at adapter_dim
        # and learned projection layers map between dim_input and adapter_dim.
        core_dim = adapter_dim if adapter_dim is not None else dim_input

        if adapter_dim is not None:
            self.input_adapter = nn.Linear(dim_input, adapter_dim)
            if output_adapter_hidden:
                # A rank-(adapter_dim) linear output cannot represent the full
                # dim_input structure; a small MLP breaks this bottleneck while
                # keeping the cheaper core. Default None keeps the plain Linear
                # so existing checkpoints load unchanged.
                output_layers = []
                prev_dim = adapter_dim
                for h_dim in output_adapter_hidden:
                    output_layers.extend([nn.Linear(prev_dim, h_dim), nn.GELU()])
                    prev_dim = h_dim
                output_layers.append(nn.Linear(prev_dim, dim_input))
                self.output_adapter = nn.Sequential(*output_layers)
            else:
                self.output_adapter = nn.Linear(adapter_dim, dim_input)
        else:
            self.input_adapter = None
            self.output_adapter = None

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim_time), nn.SiLU(), nn.Linear(dim_time, dim_time)
        )

        # Condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(dim_condition, dim_time), nn.SiLU(), nn.Linear(dim_time, dim_time)
        )

        # Main network: input is [x, time_embed, condition_embed]
        input_dim = core_dim + dim_time + dim_time
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.SiLU(),
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, core_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x, t, condition):
        """
        Args:
            x: (B, dim_input) - current state
            t: (B, 1) - time
            condition: (B, dim_condition) - [energy, sampling_fraction, (n_layers), (dir_x, dir_y, dir_z)]
        Returns:
            v: (B, dim_input) - velocity
        """
        t_emb = self.time_embed(t)
        c_emb = self.condition_embed(condition)

        if self.input_adapter is not None:
            x = self.input_adapter(x)

        # Concatenate
        h = torch.cat([x, t_emb, c_emb], dim=-1)

        # Predict velocity
        v = self.network(h)

        if self.output_adapter is not None:
            v = self.output_adapter(v)
        return v


def build_model_from_config(config, device):
    """Build a FullyConnected from a run config's ``model`` block and move it to
    ``device``.

    Central builder for the eval / generation paths, which all construct the
    model from config with the same kwargs. Callers that build the model
    differently (e.g. omitting ``dropout`` to use the class default) are left
    to construct it inline on purpose.
    """
    m = config["model"]
    return FullyConnected(
        dim_input=m["dim_input"],
        dim_condition=m["dim_condition"],
        dim_time=m["dim_time"],
        hidden_dims=m["hidden_dims"],
        dropout=m.get("dropout", 0.0),
        adapter_dim=m.get("adapter_dim", None),
        output_adapter_hidden=m.get("output_adapter_hidden"),
    ).to(device)


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
