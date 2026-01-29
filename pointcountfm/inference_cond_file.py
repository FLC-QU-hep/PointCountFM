import argparse
import time
from collections.abc import Iterable

import showerdata
import torch


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Inference on a model")
    parser.add_argument("model", type=str, help="Model file")
    parser.add_argument(
        "input_file", type=str, help="Input file with conditioning data"
    )
    parser.add_argument(
        "--pdg-codes",
        type=int,
        nargs="+",
        default=[11, -11, 22, 130, 211, -211, 321, -321, 2112, -2112, 2212, -2212],
        help="List of PDG codes corresponding to particle classes.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help=(
            "Output file with generated samples. If not specified, points"
            "per layer will be added to the input file."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Whether to overwrite the output file if it exists."
            "If input and output files are the same, this flag"
            "determines whether to overwrite the dataset within the"
            "file if it exists."
        ),
    )
    return parser.parse_args(args)


def to_labels(pdg_codes: torch.Tensor, pdgs: Iterable[int]) -> torch.Tensor:
    labels = torch.full(pdg_codes.shape, -1, dtype=torch.int64)
    for label, pdg in enumerate(pdgs):
        labels[pdg_codes == pdg] = label
    return labels


def main():
    args = parse_args()
    torch.set_num_threads(1)

    print("Loading model...")
    start = time.time()
    model = torch.jit.load(args.model)
    model.eval()
    print(f"Model loaded in {(time.time() - start) * 1000.0:.1f}ms")

    print("Compiling inference function...")
    start = time.time()

    class InferenceModel(torch.nn.Module):
        def __init__(self, model: torch.jit.ScriptModule):
            super().__init__()
            self.model = model

        def forward(self, conditions: torch.Tensor) -> torch.Tensor:
            results = []
            for i in range(conditions.size(0)):
                result = self.model(conditions[[i]])
                results.append(result)
            return torch.cat(results, dim=0)

    inference = InferenceModel(model)
    inference = inference.to(torch.float32)
    inference = torch.jit.script(inference)
    # inference = torch.jit.optimize_for_inference(inference)
    print(f"Inference function compiled in {(time.time() - start) * 1000.0:.1f}ms")

    print("Warming up...")
    start = time.time()
    example_conditions = torch.concatenate(
        (
            torch.tensor([[50.0]]),
            torch.nn.functional.one_hot(
                torch.tensor([0]), num_classes=len(args.pdg_codes)
            ).to(torch.float32),
            torch.tensor([[0.0, 0.0, 1.0]]),
        ),
        dim=1,
    )
    inference(example_conditions)
    inference(example_conditions)
    print(f"Warmup done in {(time.time() - start) * 1000.0:.1f}ms")

    print("Loading conditioning data...")
    start = time.time()
    input_len = showerdata.get_file_length(args.input_file)
    cond_data = showerdata.load_inc_particles(args.input_file, start=input_len - 50000)
    conditions = torch.concatenate(
        (
            torch.from_numpy(cond_data.energies).to(torch.float32),
            torch.nn.functional.one_hot(
                to_labels(torch.from_numpy(cond_data.pdg), args.pdg_codes),
                num_classes=len(args.pdg_codes),
            ).to(torch.float32),
            torch.from_numpy(cond_data.directions).to(torch.float32),
        ),
        dim=1,
    )
    print(f"Conditioning data loaded in {(time.time() - start) * 1000.0:.1f}ms")

    print("Running inference...")
    start = time.time()
    results = inference(conditions)
    print(f"Inference done in {(time.time() - start):.2f}s")
    print(
        f"Average inference time: {(time.time() - start) / len(conditions) * 1000.0:.1f}ms"
    )

    print("Saving results...")
    results = (torch.clamp(results, min=0.0) + 0.5).to(torch.int32)
    start = time.time()
    output_file = args.output if args.output else args.input_file
    if output_file != args.input_file:
        showerdata.save(cond_data, output_file, overwrite=args.overwrite)
    showerdata.observables.save_observables_to_file(
        output_file,
        {
            "num_points_per_layer": results.numpy(),
        },
        overwrite=args.overwrite,
    )
    print(f"Results saved in {(time.time() - start) * 1000.0:.1f}ms")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
