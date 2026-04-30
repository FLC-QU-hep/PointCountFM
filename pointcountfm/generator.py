import argparse
from collections.abc import Iterable
from pathlib import Path

import showerdata
import torch
import torch.nn.functional as F
import yaml
from rich.progress import track
from torch import nn

from pointcountfm import flow_matching, models
from pointcountfm.preprocessing import Transformation, compose


class Sampler(nn.Module):
    def __init__(
        self,
        model: flow_matching.CNF,
        transform_inc: Transformation,
        transform_num_points: Transformation,
        dim_data: int,
        steps: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.transform_inc = transform_inc
        self.transform_num_points = transform_num_points
        self.dim_data = dim_data
        self.steps = steps

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        condition = torch.clone(condition)
        condition[:, :1] = self.transform_inc(condition[:, :1])
        samples = self.model.sample(
            (condition.shape[0], self.dim_data), self.steps, condition=condition
        )
        samples = self.transform_num_points.inverse(samples)
        return samples


def init_sampler(results_dir: Path, steps: int = 50) -> Sampler:
    config = yaml.safe_load((results_dir / "conf.yaml").read_text())
    inc_energy_tranfo = compose(config["data"]["transform_inc"])
    num_points_tranfo = compose(config["data"]["transform_num_points"])
    model_class = config["model"].pop("name")
    if model_class == "FullyConnected":
        model = models.FullyConnected(**config["model"])
    elif model_class == "ConcatSquash":
        model = models.ConcatSquash(**config["model"])
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    cnf = flow_matching.CNF(model)
    sampler = Sampler(
        cnf,
        transform_inc=inc_energy_tranfo,
        transform_num_points=num_points_tranfo,
        dim_data=config["model"]["dim_input"],
        steps=steps,
    )
    state_dict = torch.load(results_dir / "model.pt", map_location="cpu")
    sampler.load_state_dict(state_dict)
    sampler.eval()
    return sampler


def to_labels(pdg_codes: torch.Tensor, pdgs: Iterable[int]) -> torch.Tensor:
    labels = torch.full(pdg_codes.shape, -1, dtype=torch.int64)
    for label, pdg in enumerate(pdgs):
        labels[pdg_codes == pdg] = label
    return labels


def prepare_conditions(
    inc_particles: showerdata.IncidentParticles,
    pdgs: list[int] | tuple[int],
    correct_directions: bool = True,
) -> torch.Tensor:
    cond_list = [torch.as_tensor(inc_particles.energies).reshape(-1, 1)]
    labels = to_labels(torch.as_tensor(inc_particles.pdg), pdgs)
    one_hot = F.one_hot(labels, num_classes=len(pdgs)).to(torch.float32)
    cond_list.append(one_hot)
    cond_list.append(torch.as_tensor(inc_particles.directions))
    result = torch.concatenate(cond_list, dim=1)
    if correct_directions:
        result[:, -3:-1] *= -1.0
    return result


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from a trained PointCountFM model."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the directory containing the trained model and its config.",
    )
    parser.add_argument(
        "condition_file",
        type=Path,
        help="Path to the .h5 file containing the conditions to sample for.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results/samples.h5"),
        help="Where to save the generated samples (default: results/samples.h5).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of ode solver steps to use when sampling (default: 32).",
    )
    parser.add_argument(
        "--pdg-codes",
        type=int,
        nargs="+",
        default=[
            11,
            -11,
            22,
            130,
            211,
            -211,
            321,
            -321,
            2112,
            -2112,
            2212,
            -2212,
        ],
        help="List of PDG codes corresponding to particle classes (default: AllShowers list).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size to use when sampling (default: 1024).",
    )
    return parser.parse_args(args)


@torch.inference_mode()
def main(args: list[str] | None = None) -> None:
    parsed_args = parse_args(args)
    torch.set_num_threads(2)
    sampler = init_sampler(parsed_args.results_dir, steps=parsed_args.steps)
    sampler.compile(fullgraph=True, dynamic=True)
    inc_particles = showerdata.load_inc_particles(parsed_args.condition_file)
    conditions = prepare_conditions(inc_particles, parsed_args.pdg_codes)
    all_samples = []
    for start in track(
        range(0, conditions.shape[0], parsed_args.batch_size), description="Sampling"
    ):
        end = min(start + parsed_args.batch_size, conditions.shape[0])
        all_samples.append(sampler(conditions[start:end]))
    samples = torch.concatenate(all_samples, dim=0)
    showerdata.save(inc_particles, parsed_args.output)
    showerdata.observables.save_observables_to_file(
        parsed_args.output,
        {"num_points_per_layer": samples.cpu().numpy()},
    )


if __name__ == "__main__":
    main()
