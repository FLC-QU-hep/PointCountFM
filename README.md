# Transferable Fast Calorimeter Shower Generation via Multi-Geometry Pre-training

[![arXiv](https://img.shields.io/badge/arXiv-2608.18233-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2608.18233)
[![Python Version](https://img.shields.io/badge/Python_3.12-306998?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch_2.6-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/FLC-QU-hep/PointCountFM?tab=MIT-1-ov-file)

PointCountFM release (`multi-geometry` branch) for the paper
**Transferable Fast Calorimeter Shower Generation via Multi-Geometry Pre-training**
(T. Buss, H. Day-Hall, F. Gaede, G. Kasieczka, K. Krüger, P. McKeown, L. Valente).
PointCountFM is a conditional flow matching model that generates the number of
points per calorimeter layer, conditioned on incident energy, sampling fraction
and number of active layers. In the paper's pipeline it is the condition
producer: its per-layer counts condition the
[AllShowers](https://github.com/FLC-QU-hep/AllShowers/tree/multi-geometry)
point cloud model (`multi-geometry` branch there as well).

The multi-geometry datasets are published in a single research-data record
([DOI 10.25592/uhhfdm.19103](https://doi.org/10.25592/uhhfdm.19103)) and the
pre-trained weights on Hugging Face
([FLC-QU-hep/PointCountFM-multi-geometry](https://huggingface.co/FLC-QU-hep/PointCountFM-multi-geometry)).
The configurations of the paper's pre-trainings and fine-tunings are in `config/`.

## Requirements
- pytorch: for training and inference of ML models
- numpy: only as input/output data format
- matplotlib: for visualization of data
- h5py: for reading training data and saving generated data
- PyYAML: for reading configuration files

## Setup
To install the python requirements, you can use either pip or conda. Choose the method that fits your environment best. Only the pip setup has been tested properly. The C++ setup is only required if you want to run the C++ inference code.

### Python (with conda)
```bash
conda env create -f environment.yml
conda activate fastshowerflow
```

### Python (with pip)
```bash
# module load maxwell python/3.12
python -m venv venv
source venv/bin/activate
pip install -r requirements/dev.txt
```
On Maxwell, use `module load` to load python 3.12. If you do not need the development packages, use `requirements.txt` instead of `requirements/dev.txt`.

### C++ (optional)
```bash
# on Maxwell HPC: module load maxwell gcc/12.2 cmake/3.28.3 hdf5/1.14.3

mkdir lib && cd lib
curl -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip
unzip libtorch.zip && cd ..

mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="../lib/libtorch" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      ../cpp
make
```

## Data
The training data is an HDF5 file containing:
- `energy`: incident energy of the particle, shape `(n_showers, 1)`
- `num_points`: number of points per layer, shape `(n_showers, max_layers)`
- `sampling_fraction`: calorimeter sampling fraction, shape `(n_showers, 1)`
- `n_layers`: number of active layers per shower, shape `(n_showers, 1)`

`max_layers` must match `model/dim_input` in the configuration file.

## Training Workflow

### Step 1: Preprocess the data
Extract the PCFM training data (per-layer counts + conditioning scalars) from
the point cloud files of the research-data record
([DOI 10.25592/uhhfdm.19103](https://doi.org/10.25592/uhhfdm.19103)), placed
under `data/`:

```bash
python src/preprocess_lemurs.py                          # LEMURS pre-training pool
python src/preprocess_simplebox.py --input ... --output ...   # SimpleBox pool
python src/preprocess_allegro.py                         # FCCee-ALLEGRO target
```

Each writes `energy`, `num_points`, `sampling_fraction`, and `n_layers` to a
per-detector training file under `data/` (e.g. `data/LEMURS_pretraining_4M.h5`).
See `--help` of each script for the input/output options. The per-detector
fine-tuning files referenced by the configs (e.g. `data/cld_finetune_150k.h5`)
follow the same schema and are extracted the same way from the corresponding
record files.

### Step 2: Train the model
```bash
python src/trainer.py config/pretrain/lemurs.yaml
```

Trains the FullyConnected flow matching model using the matching config under `config/` (see the Configuration section for the layout). Results are saved per study, then per run: `results/<study>/<run>/`.

### Step 3: Generate condition files for AllShowers
```bash
python src/compute_bias_correction.py --result_dir results/<study>/<run>   # per-layer bias-correction fit
python src/generate_pcfm_cond.py --downstream <name> ...                   # writes the condition .h5
```

`generate_pcfm_cond.py` samples per-layer point counts from a trained PCFM run
(preferring `best_physics_model.pt`), applies the fitted bias correction, and
writes the condition file that the AllShowers `multi-geometry` model
consumes. Its built-in per-detector defaults assume the internal run layout.
Point it at your own runs and files with `--pcfm-base`, `--cond-file` and
`--out-dir` (see `--help`).

---

## Usage
The main entry point is `src/trainer.py`:
```bash
python src/trainer.py [options] config/pretrain/lemurs.yaml
```

Results are saved to `results/<study>/<run>/` as configured.

### Options
| Option          | Short | Description                                             |
|-----------------|-------|---------------------------------------------------------|
| `--help`        | `-h`  | Show the help message                                   |
| `--device`      | `-d`  | Device to use (`cpu`, `cuda`), auto-selected if omitted |
| `--fast-dev-run`|       | Run a fast development run (2 epochs, 1000 samples)     |
| `--no-comet`    |       | Disable Comet ML tracking                               |

### Training Outputs
```
results/<study>/<run>/
  checkpoint.pt          # latest checkpoint (model, optimizer, scheduler, losses)
  best_model.pt          # best model weights (lowest validation loss)
  best_physics_model.pt  # best physics score (mean |total-hits bias %|)
  snapshots/             # periodic epoch_XXXXX.pt snapshots (finetune runs)
  losses.csv             # epoch, train_loss, val_loss
  conf.yaml              # copy of the config used
  plots/
    loss_vs_epoch.pdf    # train + validation loss (updated every epoch)
    lr_vs_step.pdf       # learning rate schedule (updated every epoch)
    test/
      epoch_50/          # every test_every epochs (fixed-geometry targets)
        layer_hists.pdf  # per-layer count histograms with ratio panels
      epoch_100/
        ...
```

### C++ Inference
```bash
./build/inference results/<run>/compiled.pt results/<run>/cpp_samples.h5 [n_samples]
```
The C++ path expects a TorchScript export (`compiled.pt`) of the trained model.
The export step is not part of this release.

## Configuration
`config/` ships the recipe of every training of the multi-geometry paper,
organised by study:
- `config/pretrain/`: the three pre-trainings (`simplebox.yaml`, `lemurs.yaml`,
  `simplebox_mini.yaml`)
- `config/allegro_finetuning/`: the transfer to FCCee-ALLEGRO, one folder per
  arm (`from_simplebox/`, `from_lemurs/`, `from_simplebox_mini/`,
  `from_scratch/`), each at the four dataset sizes `D100.yaml` to `D100k.yaml`
- `config/{cld,odd,par04_scipb,par04_siw}_finetuning/` and
  `config/simplebox_finetuning/`: the SimpleBox-pretrained fine-tuning
  (`D*.yaml`) and the from-scratch baseline (`scratch_D*.yaml`) on the other
  targets (the SimpleBox held-out top size is `D90k`).

Every file is the as-run recipe of the corresponding paper run, with paths made
repo-relative. Fine-tuning configs point to a pre-training checkpoint in
`training.pretrain_weights`. Set it to your own pre-training run or to the
released weights. The detector `scratch_D*.yaml` baselines carry no pre-trained
weights, but they do read the pre-training checkpoint for the input transforms
(`training.pretrain_transforms_from`), matching the paper's like-for-like
normalisation. That checkpoint file must exist for them too. In the paper the multi-seed
ALLEGRO bands use these same recipes with `data.data_offset = seed * D` for
the seed variations (fixed data and `training.seed` at `D100k`).

The configuration file is a YAML with the following keys: `name`, `data`, `model`, `training`.

### model
| Key             | Type   | Description                                             |
|-----------------|--------|---------------------------------------------------------|
| `name`          | string | Model class (`FullyConnected` or `ConcatSquash`)        |
| `dim_input`     | int    | Dimension of the input data (max_layers)                |
| `dim_condition` | int    | Dimension of the condition vector                       |
| `dim_time`      | int    | Dimension of the time embedding                         |
| `hidden_dims`   | list   | Hidden layer dimensions                                 |

### data
| Key                      | Type   | Description                                             |
|--------------------------|--------|---------------------------------------------------------|
| `data_file`              | string | Path to the training HDF5 file                          |
| `batch_size`             | int    | Training batch size                                     |
| `batch_size_val`         | int    | Validation batch size                                   |
| `train_fraction`         | float  | Fraction of data used for training (default: 0.975)     |
| `max_samples`            | int    | Max samples to load (null for all)                      |
| `use_nlayers_conditioning`| bool  | Include n_layers in the condition vector                |
| `transform_num_points`   | list   | Preprocessing transforms for num_points                 |
| `transform_fsamp`        | list   | Preprocessing transforms for sampling_fraction          |
| `transform_nlayers`      | list   | Preprocessing transforms for n_layers                   |

### training
| Key             | Type   | Description                                             |
|-----------------|--------|---------------------------------------------------------|
| `epochs`        | int    | Number of training epochs                               |
| `test_every`    | int    | Generate test plots every N epochs                      |
| `optimizer`     | dict   | Optimizer config (`name` + hyperparameters)              |
| `scheduler`     | dict   | LR scheduler config (`name` + hyperparameters, optional)|

If you use `OneCycleLR` or `CosineAnnealingLR` as a scheduler, the total number of steps is calculated automatically.

For an example, see `config/pretrain/lemurs.yaml`.

## Pre-commit
```bash
# pip install pre-commit  # already in dev.txt
pre-commit install
pre-commit run --all-files
```

---
For questions/comments about the code contact: [thorsten.buss@uni-hamburg.de](mailto:thorsten.buss@uni-hamburg.de)<br/>
For questions about this `multi-geometry` branch contact: [lorenzo.valente@uni-hamburg.de](mailto:lorenzo.valente@uni-hamburg.de)

The `multi-geometry` branch was written for the paper:

**Transferable Fast Calorimeter Shower Generation via Multi-Geometry Pre-training**<br/>
[https://arxiv.org/abs/2608.18233](https://arxiv.org/abs/2608.18233)<br/>
*Thorsten Buss, Henry Day-Hall, Frank Gaede, Gregor Kasieczka, Katja Krüger, Peter McKeown and Lorenzo Valente*

PointCountFM was introduced with the AllShowers backbone in:

**AllShowers: One model for all calorimeter showers**<br/>
[https://arxiv.org/abs/2601.11716](https://arxiv.org/abs/2601.11716)<br/>
*Thorsten Buss, Henry Day-Hall, Frank Gaede, Gregor Kasieczka and Katja Krüger*
