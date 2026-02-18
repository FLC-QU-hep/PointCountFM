import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt
import showerdata


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert showerdata to a simpler format"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input HDF5 file containing showerdata",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to the output HDF5 file to save the converted data",
    )
    parser.add_argument(
        "-l",
        "--label_list",
        type=int,
        nargs="+",
        default=None,
        help="List of PDG codes to convert to labels (default: all unique PDG codes in the data)",
    )
    return parser.parse_args(args)


def to_labels(
    pdg: npt.NDArray[np.integer], label_list: Iterable[int]
) -> npt.NDArray[np.int64]:
    pdg = pdg.reshape(-1)
    labels = np.zeros(pdg.shape, dtype=np.int64)
    for local_label, local_pdg in enumerate(label_list):
        labels[pdg == local_pdg] = local_label
    return labels


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_arguments(args)

    data = showerdata.load_inc_particles(parsed_args.input_file)
    try:
        num_points = showerdata.observables.read_observables_from_file(
            parsed_args.input_file, ["num_points_per_layer"]
        )["num_points_per_layer"]
    except ValueError as e:
        if "not found in file" in str(e):
            print(
                "'num_points_per_layer' could not be found in the input file. Having "
                "you forgotten to run 'showerdata add-observables' on the input file?"
                "No output written."
            )
            sys.exit(1)
        else:
            raise e

    print(f"Loaded {len(data)} events from {parsed_args.input_file}")

    if parsed_args.label_list is not None:
        label_list: list[int] = parsed_args.label_list
    else:
        label_list: list[int] = np.unique(data.pdg).tolist()
    label_list.sort(key=lambda x: (abs(x), -x))
    print(f"Using label list: {label_list}")

    labels = to_labels(
        pdg=data.pdg,
        label_list=label_list,
    )

    with h5py.File(parsed_args.output_file, "w") as file:
        file.create_dataset("directions", data=data.directions)
        file.create_dataset("energies", data=data.energies)
        file.create_dataset("labels", data=labels)
        file.create_dataset("num_points", data=num_points)
    print(f"Saved converted data to {parsed_args.output_file}")


if __name__ == "__main__":
    main()
