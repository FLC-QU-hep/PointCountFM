# PointCountFM
A conditional flow matching model to generate the number of points per layer in a particle shower. The model can be used as part of a generative model for particle showers. It generates the number of points per layer in a particle shower given the incident energy.

## requirements
- pytorch: for training and inference of ML models
- numpy: only as input/output data format
- matplotlib: for visualization of data
- h5py: for reading training data and saving generated data
- PyYAML: for reading configuration files

## setup
To install the python requirements, you can use either pip or conda. Choose the method that fits your environment best. Only the pip setup has been tested properly. The C++ setup is only required if you want to run the C++ inference code.

### python (with conda)
With conda, you can install the required packages by running:
```bash
# make sure to have conda installed and loaded
conda env create -f environment.yml
conda activate fastshowerflow
```
This will create a new conda environment called `fastshowerflow` and install all required packages. All necessary development packages are included in `environment.yml` file.

### python (with pip)
Use this instead of conda if you do not want to use conda. With pip, you can install the required packages by running:
```bash
# module load maxwell python/3.12
python -m venv venv
source venv/bin/activate
pip install -r requirements/dev.txt
```
On Maxwell, you can use the provided `module load` command to load python 3.12. Make sure not to have loaded any conflicting modules. This will create a new virtual environment called `venv` and install all required packages. If do not need the development packages, you can use `requirements.txt` instead of `requirements/dev.txt`.

### c++ (optional)
The C++ code is compiled using CMake. To compile the code, run the following commands:
```bash
# make sure to have cmake, hdf5, and a C++ compiler available
# on Maxwell HPC, you can uncomment the following command to load the required modules:
#module load maxwell gcc/12.2 cmake/3.28.3 hdf5/1.14.3

# manually install libtorch
mkdir lib
cd lib
curl -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip
unzip libtorch.zip
cd ..

# create a build directory and compile the code
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="../lib/libtorch" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ../cpp
make
```

## data
On default, the training data is expected to be in `data/pions.h5`. This file contains the following datasets:
- `energy`: the incident energy of the particle (shape: `(n_showers,1)
- `num_points`: the number of points per layer (shape: `(n_showers,n_layers)`)
`num_layers` must be the same as `model/dim_input` in the configuration file.

## usage
The main entry point is the `src/trainer.py` script. It can be run with the following command:
```bash
python src/trainer.py [options] config/config.yaml
```
The configuration file `config/config.yaml` specifies all hyperparameters, preprocessing steps, and the training data. The script will train a model and save it to `results/%Y%m%d_%H%M%S_name/` where `name` is the name specified in the configuration file and `%Y%m%d_%H%M%S` is the current date and time. It also generates 10000 samples and saves them to `results/%Y%m%d_%H%M%S_name/n_samples.h5`.

### options
| Option          | Short | Description                                             |
|-----------------|-------|---------------------------------------------------------|
| `--help`        | `-h`  | Show the help message                                   |
| `--device`      | `-d`  | The device to run the model on (`cpu` or `cuda`) (if not specified it will be automatically selected) |
| `--time`        | `-t`  | Run a timing test on the model                          |
| `--fast-dev-run`|       | Run a fast development run                              |


### cpp inference
The C++ code can be run with the following command:
```bash
./build/inference results/%Y%m%d_%H%M%S_name/compiled.pt results/%Y%m%d_%H%M%S_name/cpp_samples.h5 [n_samples]
```
This will load the model from `results/%Y%m%d_%H%M%S_name/compiled.pt` and generate `n_samples` samples. If `n_samples` is not specified, it will generate 100 samples. The samples are saved to `results/%Y%m%d_%H%M%S_name/cpp_samples.h5`.

## configuration
The configuration file is a YAML with the following keys:
- `model`: specifies the model architecture and hyperparameters
- `data`: specifies the training data and preprocessing steps
- `training`: specifies the training hyperparameters
- `name`: a descriptive name for the run

### model
| Key             | Type   | Description                                             |
|-----------------|--------|---------------------------------------------------------|
| `name`          | string | The model class (`FullyConnected` or `ConcatSquash`)    |
| `dim_input`     | int    | The dimension of the input data                         |
| `dim_condition` | int    | The dimension of the condition                          |
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
| `optimizer`     | dict   | The optimizer (`name` key) and its hyperparameters      |
| `scheduler`     | dict   | The learning rate scheduler (`name` key) and its hyperparameters (optional) |

If you use OneCycleLR or CosineAnnealing as a scheduler, the maximum number of iterations is calculated automatically.

For an example configuration file, see `config/config.yaml`.

## pre-commit
This repository uses [pre-commit](https://pre-commit.com) to run checks on the code before committing. To install pre-commit, run:
```bash
# pip install pre-commit  # already in the def.txt requirements file
pre-commit install
```
This will install pre-commit and set up the checks. If you want to run the checks manually, you can run:
```bash
pre-commit run --all-files
```
This will run all checks on all files.

---
If you have any questions or comments about this repository, please contact [thorsten.buss@uni-hamburg.de](mailto:thorsten.buss@uni-hamburg.de).
