import argparse
import copy
import datetime
import os
import shutil
import sys
import time
import warnings

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from pointcountfm import flow_matching, models
from pointcountfm.data_loader import DataLoader, get_loaders
from pointcountfm.preprocessing import Transformation


class Trainer:
    def __init__(
        self,
        model_config: dict,
        data_config: dict,
        training_config: dict,
        device: torch.device,
        result_dir: str,
    ) -> None:
        self.device = device
        self.result_dir = result_dir
        self.epochs = training_config["epochs"]

        data_config["device"] = self.device
        self.train_loader, self.val_loader = get_loaders(**data_config)
        self.data_config = data_config
        self.dim_data = self.train_loader.data.shape[1]
        self.dim_condition = self.train_loader.condition.shape[1]
        self.steps = training_config.get("steps", 50)

        self.model = self.__init_model(model_config)
        self.optimizer = self.__init_optimizer(training_config["optimizer"])
        self.scheduler = self.__init_scheduler(training_config.get("scheduler", {}))

        self.train_losses = []
        self.val_losses = []

        self.train_losses_distill = []
        self.val_losses_distill = []

        self.loss_log = self.__get_file_path("losses.csv")
        self.loss_plot = self.__get_file_path("losses.pdf")
        self.model_path = self.__get_file_path("model.pt")
        self.checkpoint_path = self.__get_file_path("checkpoint.pt")
        self.new_samples_path = self.__get_file_path("new_samples.h5")
        self.compiled_path = self.__get_file_path("compiled.pt")

        self.loss_log_distill = self.__get_file_path("losses_distill.csv")
        self.loss_plot_distill = self.__get_file_path("losses_distill.pdf")
        self.model_path_distill = self.__get_file_path("model_distill.pt")
        self.checkpoint_path_distill = self.__get_file_path("checkpoint_distill.pt")
        self.new_samples_path_distilled = self.__get_file_path(
            "new_samples_distilled.h5"
        )

        if os.path.exists(self.checkpoint_path):
            self.__load_checkpoint()

        if os.path.exists(self.checkpoint_path_distill):
            self.__init_distill()
            self.__load_checkpoint_distill()
        else:
            self.model_distill = None
            self.optimizer_distill = None
            self.scheduler_distill = None
            self.train_loader_distill = None
            self.val_loader_distill = None

        example_inputs = torch.ones(1, 1, dtype=torch.float32) * 50.0
        if self.train_loader.num_classes > 0:
            example_inputs = torch.cat(
                (
                    example_inputs,
                    torch.zeros(1, self.train_loader.num_classes, dtype=torch.float32),
                ),
                dim=1,
            )
            example_inputs[:, 1] = 1.0  # Set one class to 1.0
        if self.dim_condition > example_inputs.shape[1]:
            example_inputs = torch.cat(
                (
                    example_inputs,
                    torch.zeros(
                        1,
                        self.dim_condition - example_inputs.shape[1],
                        dtype=torch.float32,
                    ),
                ),
                dim=1,
            )
            example_inputs[:, -1] = 1.0
        self.example_inputs = example_inputs

        print("Trainer initialized.")
        print(f"Device: {self.device}")
        print(f"num_train: {self.train_loader.data.shape[0]}")
        print(f"num_val: {self.val_loader.data.shape[0]}")
        print(f"dim_data: {self.dim_data}")
        print(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        print()
        sys.stdout.flush()

    def __init_model(self, config: dict) -> flow_matching.CNF:
        config = config.copy()
        if "name" not in config:
            raise ValueError("Model configuration missing.")
        flow_config = config.pop("flow") if "flow" in config else {}
        model_name = config.pop("name")
        model_class = getattr(models, model_name)
        model_object = model_class(**config)
        flow = flow_matching.CNF(model_object, **flow_config)
        return flow.to(self.device)

    def __init_optimizer(self, config: dict) -> optim.Optimizer:
        config = config.copy()
        optimizer_name = config.pop("name")
        optimizer_class = getattr(optim, optimizer_name)
        if "lr" in config:
            self.lr = config["lr"]
        else:
            raise ValueError("Learning rate missing.")
        return optimizer_class(self.model.parameters(), **config)

    def __init_scheduler(self, config: dict) -> optim.lr_scheduler._LRScheduler | None:
        config = config.copy()
        if "name" not in config:
            return None
        scheduler_name = config.pop("name")
        if scheduler_name not in optim.lr_scheduler.__all__:
            raise ValueError(
                f"Scheduler {scheduler_name} not found in torch.optim.lr_scheduler."
            )
        scheduler_class: type[optim.lr_scheduler._LRScheduler] = getattr(
            optim.lr_scheduler, scheduler_name
        )
        if scheduler_class is optim.lr_scheduler.OneCycleLR:
            config["max_lr"] = self.lr
            config["total_steps"] = len(self.train_loader) * self.epochs
        elif scheduler_class is optim.lr_scheduler.CosineAnnealingLR:
            config["T_max"] = len(self.train_loader) * self.epochs
        return scheduler_class(self.optimizer, **config)

    def __get_file_path(self, filename: str) -> str:
        full_path = os.path.join(self.result_dir, filename)
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return full_path

    def __save_checkpoint(self) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_losses": torch.tensor(self.train_losses),
            "val_losses": torch.tensor(self.val_losses),
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        torch.save(checkpoint, self.checkpoint_path)

    def __save_checkpoint_distill(self) -> None:
        if self.model_distill is None or self.optimizer_distill is None:
            raise ValueError("Distilled model or optimizer not initialized.")
        checkpoint = {
            "model": self.model_distill.state_dict(),
            "optimizer": self.optimizer_distill.state_dict(),
            "train_losses": torch.tensor(self.train_losses_distill),
            "val_losses": torch.tensor(self.val_losses_distill),
        }
        if self.scheduler_distill is not None:
            checkpoint["scheduler"] = self.scheduler_distill.state_dict()
        torch.save(checkpoint, self.checkpoint_path_distill)

    def __load_checkpoint(self) -> None:
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_losses = checkpoint["train_losses"].tolist()
        self.val_losses = checkpoint["val_losses"].tolist()
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def __load_checkpoint_distill(self) -> None:
        if self.model_distill is None or self.optimizer_distill is None:
            raise ValueError("Distilled model or optimizer not initialized.")
        checkpoint = torch.load(
            self.checkpoint_path_distill,
            map_location=self.device,
            weights_only=True,
        )
        self.model_distill.load_state_dict(checkpoint["model"])
        self.optimizer_distill.load_state_dict(checkpoint["optimizer"])
        self.train_losses_distill = checkpoint["train_losses"].tolist()
        self.val_losses_distill = checkpoint["val_losses"].tolist()
        if self.scheduler_distill is not None:
            self.scheduler_distill.load_state_dict(checkpoint["scheduler"])

    def __save_losses(self, distill: bool = False) -> None:
        file_name = self.loss_log_distill if distill else self.loss_log
        train_losses = self.train_losses_distill if distill else self.train_losses
        val_losses = self.val_losses_distill if distill else self.val_losses

        with open(file_name, "w") as file:
            for epoch, (train_loss, val_loss) in enumerate(
                zip(train_losses, val_losses)
            ):
                file.write(f"{epoch + 1} {train_loss} {val_loss}\n")

    def __plot_losses(self, distill: bool = False) -> None:
        loss_plot = self.loss_plot_distill if distill else self.loss_plot
        train_losses = self.train_losses_distill if distill else self.train_losses
        val_losses = self.val_losses_distill if distill else self.val_losses

        plt.plot(train_losses, label="Train loss")
        plt.plot(val_losses, label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(loss_plot)
        plt.close()

    def train(self) -> None:
        print("Training started.")
        sys.stdout.flush()
        for epoch in range(len(self.train_losses), self.epochs):
            self.model.train()
            train_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                x = batch["data"]
                condition = batch["condition"]
                noise = batch["noise"]
                losses = self.model.loss(x, condition, noise)
                loss = torch.mean(losses)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                if self.scheduler is not None:
                    self.scheduler.step()
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in self.val_loader:
                    x = batch["data"]
                    condition = batch["condition"]
                    noise = batch["noise"]
                    losses = self.model.loss(x, condition, noise)
                    loss = torch.mean(losses)
                    val_loss += loss.item()
            val_loss /= len(self.val_loader)
            self.val_losses.append(val_loss)

            self.__save_checkpoint()
            self.__save_losses()
            self.__plot_losses()

            print(f"=== Epoch {epoch + 1}/{self.epochs} ===")
            print(f"Train loss: {train_loss:.4f}")
            print(f"Val loss: {val_loss:.4f}")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print()
            sys.stdout.flush()

        print("Training finished.")
        sys.stdout.flush()

    def sample_condition(
        self,
        num_samples: int,
        min_energy: float | None = None,
        max_energy: float | None = None,
        class_label: int | None = None,
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> torch.Tensor:
        if min_energy is None:
            min_energy = self.train_loader.energy_min
        if max_energy is None:
            max_energy = self.train_loader.energy_max

        incident_energy = min_energy + (max_energy - min_energy) * torch.rand(
            num_samples, 1
        )
        if self.train_loader.num_classes == 0:
            return incident_energy

        if class_label is not None:
            class_label_tensor = torch.full(
                (num_samples,), class_label, dtype=torch.int64
            )
        else:
            class_label_tensor = torch.randint(
                0, self.train_loader.num_classes, (num_samples,), dtype=torch.int64
            )
        class_label_tensor = F.one_hot(
            class_label_tensor, num_classes=self.train_loader.num_classes
        ).to(torch.get_default_dtype())
        if self.dim_condition > 1 + self.train_loader.num_classes:
            direction_tensor = torch.tensor(
                direction, dtype=torch.get_default_dtype()
            ).reshape(1, 3)
            direction_tensor = direction_tensor.repeat(num_samples, 1)
            return torch.cat(
                (incident_energy, class_label_tensor, direction_tensor), dim=1
            )
        else:
            return torch.cat((incident_energy, class_label_tensor), dim=1)

    @staticmethod
    def __get_condition(
        data_loader: DataLoader, num_samples: int | None = None
    ) -> torch.Tensor:
        if num_samples is None:
            num_samples = int(data_loader.condition.shape[0])
        if num_samples > data_loader.condition.shape[0]:
            warnings.warn(
                "Number of samples requested exceeds number of validation samples.",
                UserWarning,
            )
            condition = data_loader.condition
        else:
            condition = data_loader.condition[:num_samples]
        condition[:, :1] = data_loader.transform_inc.inverse(condition[:, :1])
        condition = condition.to("cpu", copy=True)
        if len(condition) < num_samples:
            repeats = -(-num_samples // len(condition))  # Ceiling division
            condition = condition.repeat(repeats, 1)[:num_samples]
        return condition

    def get_val_condition(self, num_samples: int | None = None) -> torch.Tensor:
        return self.__get_condition(self.val_loader, num_samples)

    def get_train_condition(self, num_samples: int | None = None) -> torch.Tensor:
        return self.__get_condition(self.train_loader, num_samples)

    def sample(
        self, condition: torch.Tensor, num_steps: int = 0, distilled: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_steps == 0:
            num_steps = self.steps
        input_device = condition.device
        condition = condition.to(self.device, copy=True)
        model = self.model_distill if distilled else self.model
        if model is None:
            raise ValueError(
                "Distilled model not initialized."
                if distilled
                else "Model not initialized."
            )
        model.eval()
        with torch.no_grad():
            condition[:, :1] = self.val_loader.transform_inc(condition[:, :1])
            samples, noise = model.sample_return_z(
                (condition.shape[0], self.dim_data), num_steps, condition
            )
            samples_t = self.val_loader.transform_num_points.inverse(samples)
        return (
            samples_t.to(input_device),
            noise.to(input_device),
            samples.to(input_device),
            condition.to(input_device),
        )

    def sample_batch(
        self,
        condition: torch.Tensor,
        batch_size: int,
        verbose: bool = False,
        num_steps: int = 0,
        distilled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if verbose:
            print("Sampling started.")
            print("Batch size:", batch_size)
            print("Number of samples:", condition.shape[0])
            print("Batch size:", batch_size)
            print("Number of batches:", -(-condition.shape[0] // batch_size))
            print("number of threads:", torch.get_num_threads())
            print("device:", self.device)
            print("dtype:", condition.dtype)
            sys.stdout.flush()
        time_start = time.time()
        cond_batches = torch.split(condition, batch_size)
        samples = []
        noise = []
        samples_raw = []
        cond_t = []
        for cond_batch in cond_batches:
            samples_l, noise_l, samples_raw_l, cond_t_l = self.sample(
                cond_batch, num_steps, distilled
            )
            samples.append(samples_l)
            noise.append(noise_l)
            samples_raw.append(samples_raw_l)
            cond_t.append(cond_t_l)
        if verbose:
            time_end = time.time()
            print(f"Sampling finished in {1000.0 * (time_end - time_start):.2f} ms.")
            print(
                f"That's {1000.0 * (time_end - time_start) / len(cond_batches):.2f} ms per batch."
            )
            print()
            sys.stdout.flush()
        return (
            torch.cat(samples, dim=0),
            torch.cat(noise, dim=0),
            torch.cat(samples_raw, dim=0),
            torch.cat(cond_t, dim=0),
        )

    def sample_and_save(
        self,
        condition: torch.Tensor,
        batch_size: int,
        verbose: bool = False,
        save_noise: bool = False,
        num_steps: int = 0,
        distilled: bool = False,
    ) -> None:
        samples, noise, samples_raw, cond_t = self.sample_batch(
            condition,
            batch_size,
            verbose=verbose,
            num_steps=num_steps,
            distilled=distilled,
        )
        energy = condition[:, 0:1]
        if self.train_loader.num_classes == 0:
            labels = None
        else:
            labels = torch.argmax(
                condition[:, 1 : self.train_loader.num_classes + 1], dim=1
            )
        directions = condition[:, -3:] if self.train_loader.has_directions else None
        file_path = (
            self.new_samples_path_distilled if distilled else self.new_samples_path
        )
        with h5py.File(file_path, "w") as file:
            file.create_dataset("energy", data=energy.numpy())
            file.create_dataset("num_points", data=samples.numpy())
            if labels is not None:
                file.create_dataset("labels", data=labels.numpy())
            if directions is not None:
                file.create_dataset("directions", data=directions.numpy())
            if save_noise:
                file.create_dataset("noise", data=noise.numpy())
                file.create_dataset("preprocessed_data", data=samples_raw.numpy())
                file.create_dataset("preprocessed_energy", data=cond_t.numpy())

    def to(self, device_dtype: torch.device | torch.dtype | str) -> None:
        self.model.to(device_dtype)
        self.val_loader.to(device_dtype)
        self.train_loader.to(device_dtype)
        if self.model_distill is not None:
            self.model_distill.to(device_dtype)
        if self.val_loader_distill is not None:
            self.val_loader_distill.to(device_dtype)
        if self.train_loader_distill is not None:
            self.train_loader_distill.to(device_dtype)
        if isinstance(device_dtype, torch.device):
            self.device = device_dtype
        elif isinstance(device_dtype, str):
            device_dtype = device_dtype.lower()
            try:
                device = torch.device(device_dtype)
                self.device = device
            except RuntimeError:
                # If the device string is not recognized, assume it is a dtype.
                pass

    @torch.inference_mode()
    def compile(self) -> None:
        self.model.eval()
        print("Compiling started.")

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

        sampler = Sampler(
            model=self.model,
            transform_inc=self.val_loader.transform_inc,
            transform_num_points=self.val_loader.transform_num_points,
            dim_data=self.dim_data,
            steps=self.steps,
        )
        sampler = copy.deepcopy(sampler)
        sampler = sampler.to("cpu")
        sampler = sampler.to(torch.float32)
        sampler = torch.jit.trace(sampler, self.example_inputs, check_trace=False)
        torch.jit.save(sampler, self.compiled_path)
        print("Compiling finished.")

    def __init_distill(self) -> None:
        if not os.path.exists(self.new_samples_path):
            print("Sampling for distillation started.")
            sys.stdout.flush()
            cond = torch.concatenate(
                [
                    self.get_train_condition(),
                    self.get_val_condition(),
                ],
                dim=0,
            )
            self.sample_and_save(
                cond, self.val_loader.batch_size, save_noise=True, num_steps=200
            )
            print("Sampling finished.")
            print()
            sys.stdout.flush()
        data_config = self.data_config.copy()
        data_config["data_file"] = self.new_samples_path
        data_config["load_noise"] = True
        self.train_loader_distill, self.val_loader_distill = get_loaders(**data_config)
        self.train_loader_distill.to(self.device)
        self.val_loader_distill.to(self.device)

        self.model_distill = flow_matching.Distilled(self.model)
        self.model_distill = self.model_distill.to(self.device)
        self.optimizer_distill = optim.Adam(self.model_distill.parameters(), lr=1e-4)
        self.scheduler_distill = optim.lr_scheduler.OneCycleLR(
            self.optimizer_distill,
            max_lr=1e-4,
            total_steps=len(self.train_loader_distill) * self.epochs,
        )

    def distill(self):
        if self.model_distill is None:
            self.__init_distill()
        if (
            self.model_distill is None
            or self.optimizer_distill is None
            or self.train_loader_distill is None
            or self.val_loader_distill is None
        ):
            raise ValueError("Distilled model initialization failed.")
        print("Distillation started.")
        sys.stdout.flush()

        for epoch in range(len(self.train_losses_distill), self.epochs):
            self.model_distill.train()
            train_loss = 0
            for batch in self.train_loader_distill:
                self.optimizer_distill.zero_grad()
                x = batch["data"]
                condition = batch["condition"]
                noise = batch["noise"]
                x_pred = self.model_distill(noise, condition)
                loss = torch.mean((x_pred - x).square())
                loss.backward()
                self.optimizer_distill.step()
                train_loss += loss.item()
                if self.scheduler_distill is not None:
                    self.scheduler_distill.step()
            train_loss /= len(self.train_loader_distill)

            self.model_distill.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in self.val_loader_distill:
                    x = batch["data"]
                    condition = batch["condition"]
                    noise = batch["noise"]
                    x_pred = self.model_distill(noise, condition)
                    loss = torch.mean((x_pred - x).square())
                    val_loss += loss.item()
            val_loss /= len(self.val_loader_distill)

            print(f"=== Epoch {epoch + 1}/{self.epochs} ===")
            print(f"Train loss: {train_loss:.4f}")
            print(f"Val loss: {val_loss:.4f}")
            print(f"Learning rate: {self.optimizer_distill.param_groups[0]['lr']:.2e}")
            print()
            sys.stdout.flush()

            self.train_losses_distill.append(train_loss)
            self.val_losses_distill.append(val_loss)

            self.__save_checkpoint_distill()
            self.__save_losses(distill=True)
            self.__plot_losses(distill=True)

    def compile_distill(self) -> None:
        if self.model_distill is None:
            raise ValueError("Distilled model not initialized.")
        self.model_distill.eval()
        print("Compiling distillation started.")

        class SamplerDistill(nn.Module):
            def __init__(
                self,
                model: flow_matching.Distilled,
                transform_inc: Transformation,
                transform_num_points: Transformation,
                dim_data: int,
            ) -> None:
                super().__init__()
                self.model = model
                self.transform_inc = transform_inc
                self.transform_num_points = transform_num_points
                self.dim_data = dim_data

            def forward(self, condition: torch.Tensor) -> torch.Tensor:
                condition = torch.clone(condition)
                condition[:, :1] = self.transform_inc(condition[:, :1])
                samples = self.model.sample(
                    (condition.shape[0], self.dim_data), condition
                )
                samples = self.transform_num_points.inverse(samples)
                return samples

        sampler = SamplerDistill(
            model=self.model_distill,
            transform_inc=self.train_loader.transform_inc,
            transform_num_points=self.train_loader.transform_num_points,
            dim_data=self.dim_data,
        )
        sampler = copy.deepcopy(sampler)
        sampler = sampler.to("cpu")
        sampler = sampler.to(torch.float32)
        sampler = torch.jit.trace(sampler, self.example_inputs, check_trace=False)
        torch.jit.save(sampler, self.compiled_path.replace(".pt", "_distill.pt"))
        print("Compiling distillation finished.")
        sys.stdout.flush()


def setup_result_path(run_name: str, conf_file: str, fast_dev_run: bool = False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)

    now = datetime.datetime.now()
    while True:
        full_run_name = now.strftime("%Y%m%d_%H%M%S") + "_" + run_name
        result_path = os.path.join(repo_dir, "results", full_run_name)
        if not os.path.exists(result_path):
            if not fast_dev_run:
                os.makedirs(result_path)
            else:
                result_path = os.path.join(repo_dir, "results/test")
                if os.path.exists(result_path):
                    shutil.rmtree(result_path)
                os.makedirs(result_path)
            break
        else:
            now += datetime.timedelta(seconds=1)

    with open(conf_file) as f:
        content_list = f.readlines()

    content_list = [line for line in content_list if not line.startswith("result_path")]
    content_list.insert(1, f"result_path: {result_path}\n")
    content = "".join(content_list)

    with open(os.path.join(result_path, "conf.yaml"), "w") as f:
        f.write(content)

    return result_path


def parse_args(args: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    parser.add_argument(
        "--fast-dev-run", action="store_true", help="Run a test with a small dataset."
    )
    parser.add_argument(
        "-d",
        "--device",
        default="",
        type=str,
        help='Device to use for training (e.g., "cpu", "cuda", "mps").',
    )
    parser.add_argument("-t", "--time", action="store_true", help="Time sampling speed")
    parser.add_argument("--distill", action="store_true", help="Distill the model")
    return parser.parse_args(args)


def main(args: list | None = None) -> None:
    mpl.use("Agg")  # Use non-interactive backend for plotting
    parsed_args = parse_args(args)
    with open(parsed_args.config) as file:
        config = yaml.safe_load(file)
    if parsed_args.device:
        device = torch.device(parsed_args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if "result_path" in config:
        result_dir = config["result_path"]
    else:
        result_dir = setup_result_path(
            config["name"], parsed_args.config, parsed_args.fast_dev_run
        )
    num_new_samples = config["training"].get("num_new_samples", 50_000)
    if parsed_args.fast_dev_run:
        config["data"]["num_train"] = 100
        config["data"]["num_val"] = 10
        config["data"]["batch_size"] = 2
        config["data"]["batch_size_val"] = 2
        config["training"]["epochs"] = 2
        num_new_samples = 10
    trainer = Trainer(
        model_config=config["model"],
        data_config=config["data"],
        training_config=config["training"],
        device=device,
        result_dir=result_dir,
    )
    trainer.train()

    if not os.path.exists(trainer.new_samples_path) and not parsed_args.distill:
        print("Sampling started.")
        sys.stdout.flush()
        cond_test = trainer.get_val_condition(num_new_samples)
        trainer.sample_and_save(cond_test, 4096, save_noise=False)
        print("Sampling finished.")
        print()
        sys.stdout.flush()

    if not os.path.exists(trainer.compiled_path):
        trainer.compile()

    if parsed_args.distill:
        trainer.distill()
        trainer.compile_distill()

    if parsed_args.distill and not os.path.exists(trainer.new_samples_path_distilled):
        print("Sampling distilled model.")
        sys.stdout.flush()
        cond_test = trainer.get_val_condition(num_new_samples)
        trainer.sample_and_save(cond_test, 4096, distilled=True)
        print("Sampling finished.")
        print()
        sys.stdout.flush()

    if parsed_args.time:
        print("Timing started.")
        sys.stdout.flush()
        torch.set_num_threads(1)
        trainer.to("cpu")
        cond_test = trainer.get_val_condition(10)
        trainer.sample_batch(cond_test, 1, verbose=True)
        if parsed_args.distill:
            trainer.sample_batch(cond_test, 1, verbose=True, distilled=True)

        torch.set_default_dtype(torch.float16)
        trainer.to(torch.float16)
        cond_test = trainer.get_val_condition(10)
        trainer.new_samples_path = os.path.join(result_dir, "new_samples_fp16.h5")
        trainer.sample_and_save(cond_test, 1, verbose=True)
        if parsed_args.distill:
            trainer.sample_batch(cond_test, 1, verbose=True, distilled=True)


if __name__ == "__main__":
    main()
