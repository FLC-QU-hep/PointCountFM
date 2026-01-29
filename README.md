# PointCountFM
[![Python Version](https://img.shields.io/badge/Python_3.13-306998?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch_2.8-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/FLC-QU-hep/PointCountFM?tab=MIT-1-ov-file)
[![Build Status](https://img.shields.io/github/actions/workflow/status/FLC-QU-hep/PointCountFM/pre_commit.yaml?label=pre-commit&logo=github)](https://github.com/FLC-QU-hep/PointCountFM/actions/workflows/pre_commit.yaml)
[![Tests](https://img.shields.io/github/actions/workflow/status/FLC-QU-hep/PointCountFM/test.yaml?label=tests&logo=github)](https://github.com/FLC-QU-hep/PointCountFM/actions/workflows/test.yaml)

A conditional flow matching model to generate the number of points per layer in a particle shower. The model can be used as part of a generative model for particle showers. It generates the number of points per layer in a particle shower given the incident particle type and kinematics.

## Table of Contents <!-- omit in toc -->
- [Requirements](#requirements)
- [Setup](#setup)
  - [Clone repository](#clone-repository)
  - [Install dependencies](#install-dependencies)
- [Data](#data)
- [Usage](#usage)
  - [options](#options)
- [configuration](#configuration)
  - [model](#model)
  - [data](#data-1)
  - [training](#training)
- [pre-commit](#pre-commit)
- [Testing](#testing)


## Requirements
- pytorch: for training and inference of ML models
- numpy: only as input/output data format
- matplotlib: for visualization of data
- h5py: for reading training data and saving generated data
- pyyaml: for reading configuration files
- showerdata: for handling calorimeter shower data

## Setup

### Clone repository
To clone the repository, run:
```bash
git clone git@github.com:FLC-QU-hep/PointCountFM.git
cd PointCountFM
```

### Install dependencies
Choose one of the following options to install the required dependencies. Only `uv` has to be tested.

#### With `uv` (option 1):
```bash
uv sync --all-groups
source .venv/bin/activate
```

#### With `pip` + `venv` (option 2):
```bash
python3.13 -m venv --prompt PointCountFM .venv
source .venv/bin/activate
pip install -e .
pip install --group dev
```
If you want to try to run the code with a different python version, you might need to adapt the `pyproject.toml` file accordingly.

#### With `conda` (option 3):
```bash
conda env create -f environment.yaml
conda activate PointCountFM
pip install -e .
```
All packages available from conda-forge will be installed via conda, the rest via pip.

## Data
You can use your own data or download the AllShowers dataset from Zenodo: [https://zenodo.org/records/18020348](https://zenodo.org/records/18020348)
To download layer level shower data (1.3 GB), run:
```bash
mkdir data
curl -o data/layer_level.h5 https://zenodo.org/records/18020348/files/layer_level.h5?download=1
```
If you want to use you own data, store it in HDF5 format with the following keys:

| key        | shape  | dtype   | description                                       |
|------------|--------|---------|---------------------------------------------------|
| directions | (n, 3) | float32 | incident particle directions                      |
| energies   | (n, 1) | float32 | incident particle energies (in any consistent unit) |
| labels     | (n)    | int32   | incident particle labels                          |
| num_points | (n, m) | int32   | number of points per layer                        |

n is the data set size and m is the number of layers in the calorimeter, which needs to be consistent with `dim_input` in the configuration file.

## Usage
The main entry point is the `pointcountfm/trainer.py` script. It can be run with the following command:
```bash
python pointcountfm/trainer.py [options] config/config.yaml
```
The configuration file `config/config.yaml` specifies all hyper-parameters, preprocessing steps, and the training data. The script will train a model and save it to `results/%Y%m%d_%H%M%S_name/` where `name` is the name specified in the configuration file and `%Y%m%d_%H%M%S` is the current date and time. It also generates 50,000 samples and saves them to `results/%Y%m%d_%H%M%S_name/new_samples.h5`.

### options
| Option          | Short | Description                                             |
|-----------------|-------|---------------------------------------------------------|
| `--help`        | `-h`  | Show the help message                                   |
| `--device`      | `-d`  | The device to run the model on (e.g. `cpu`, `mps`, or `cuda`) (if not specified it will be automatically selected) |
| `--time`        | `-t`  | Run a timing test on the model                          |
| `--fast-dev-run`|       | Run a fast development run for testing                  |

## configuration
The configuration file is a YAML with the following keys:

- `model`: specifies the model architecture and hyper-parameters
- `data`: specifies the training data and preprocessing steps
- `training`: specifies the training hyper-parameters
- `name`: a descriptive name for the run

### model
| Key             | Type   | Description                                             |
|-----------------|--------|---------------------------------------------------------|
| `name`          | string | The model class (`FullyConnected` or `ConcatSquash`)    |
| `dim_input`     | int    | The dimension of the input data                         |
| `dim_condition` | int    | The dimension of the condition (number of particle labels + 3 (directions) + 1 (energies))                      |
| `dim_time`      | int    | The dimension of the time embedding                     |
| `hidden_dims`   | list   | A list of hidden dimensions for the model               |

### data
| Key             | Type   | Description                                             |
|-----------------|--------|---------------------------------------------------------|
| `data_file`     | string | The path to the training data                           |
| `batch_size`    | int    | The batch size for training                             |
| `batch_size_val`| int    | The batch size for validation                           |
|`transform_num_points`| list | A list of the preprocessing steps for the number of points per layer (optional) |
| `transform_inc` | list   | A list of the preprocessing steps for the incident energy (optional) |

### training
| Key             | Type   | Description                                             |
|-----------------|--------|---------------------------------------------------------|
| `epochs`        | int    | The number of epochs to train the model                 |
| `optimizer`     | dict   | The optimizer (`name` key) and its hyper-parameters     |
| `scheduler`     | dict   | The learning rate scheduler (`name` key) and its hyper-parameters (optional) |

If you use OneCycleLR or CosineAnnealing as a scheduler, the maximum number of iterations is calculated automatically.

For an example configuration file, see `config/config.yaml`.

## pre-commit
This repository uses [pre-commit](https://pre-commit.com) to run checks on the code before committing. To install pre-commit, run:
```bash
pre-commit install
```
This will install pre-commit and set up the checks. If you want to run the checks manually, you can run:
```bash
pre-commit run --all-files
```
This will run all checks on all files.

## Testing
To run the unit tests, run:
```bash
python -m unittest discover -s test -p "*_test.py" -v
```

---
If you have any questions or comments about this repository, please contact [thorsten.buss@uni-hamburg.de](mailto:thorsten.buss@uni-hamburg.de).
