import os
from copy import deepcopy
from typing import Optional, Callable

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import KFold
from torch.utils.data import Subset

from omegaconf import DictConfig
from tqdm import tqdm

from modules.logger.wandb import WBLogger
from utils.plots import plot_losses
from utils.metrics import Metrics, CLASSIFICATION_TASKS


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        config: DictConfig,
        metrics: Metrics,
        save_dir: Optional[str] = None,
        save_name: Optional[str] = "model",
        n_splits: Optional[int] = 5
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_metrics = deepcopy(metrics)
        self.eval_metrics = deepcopy(metrics)
        self.logger = WBLogger(
            train_metrics=self.train_metrics,
            val_metrics=self.eval_metrics
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config
        self.save_dir = os.getcwd() if save_dir is None else save_dir
        self.save_name = save_name

        self.train_losses = []
        self.eval_losses = []

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True
        )
        self.n_splits = n_splits

    def train_step(self, train_loader: Optional[DataLoader] = None):
        self.model.train()
        train_loader = train_loader if train_loader is not None else self.train_loader
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm)
            self.optimizer.step()
            running_loss += loss.item()
            self.update_metrics(outputs, labels, mode="train")

        return running_loss / len(train_loader)

    def train(
        self,
        verbose: Optional[bool] = True,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None
    ):
        best_loss = float("inf")

        for i in tqdm(range(self.config.epochs), desc="Training"):
            self.train_metrics.reset()
            train_loss = self.train_step(train_loader)
            self.logger.log_train_epoch(num=i, loss=train_loss)
            self.train_losses.append(train_loss)

            self.eval_metrics.reset()
            eval_loss = self.eval(verbose=verbose, training=True, eval_loader=eval_loader)
            self.logger.log_val_epoch(num=i, loss=eval_loss)
            self.eval_losses.append(eval_loss)
            self.lr_scheduler.step(eval_loss)

            if verbose:
                print(f"Epoch [{i + 1}/{self.config.epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {eval_loss:.4f}")
                print(f"\nTrain metrics: {str(self.train_metrics)}")
                print(f"\nValid metrics: {str(self.eval_metrics)}")

            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save()

        return self.load_model()

    def cross_validate(self, verbose: Optional[bool] = True):
        k_fold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.config.random_seed)
        for fold, (train_idx, val_idx) in enumerate(k_fold.split(self.train_dataset)):
            print(f"\nStarting fold {fold + 1}/{self.n_splits}")
            train_subset = Subset(self.train_dataset, train_idx)
            val_subset = Subset(self.train_dataset, val_idx)
            train_loader = DataLoader(
                train_subset, batch_size=self.config.train_batch_size, shuffle=True
            )
            eval_loader = DataLoader(
                val_subset, batch_size=self.config.eval_batch_size, shuffle=False
            )
            self.train(verbose=verbose, train_loader=train_loader, eval_loader=eval_loader)

    @torch.no_grad()
    def eval(
            self,
            verbose: Optional[bool] = True,
            training: Optional[bool] = False,
            eval_loader: Optional[DataLoader] = None,
    ):
        self.model.eval()
        running_loss = 0.0
        eval_loader = eval_loader if eval_loader is not None else self.eval_loader

        for inputs, labels in eval_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            running_loss += loss.item()
            self.update_metrics(outputs, labels, mode="eval")

        if verbose and not training:
            print(f"\nValid metrics: {str(self.eval_metrics)}")

        return running_loss / len(eval_loader)

    def load_model(self) -> nn.Module:
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.save_name}.bin')))
        return self.model

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'{self.save_name}.bin'))

    def plot_losses(self):
        plot_losses(self.train_losses, self.eval_losses)

    def update_metrics(self, outputs: Tensor, labels: Tensor, mode: Optional[str] = "train"):
        metrics = self.train_metrics if mode == "train" else self.eval_metrics
        if metrics.task in CLASSIFICATION_TASKS:
            _, predicted = torch.max(outputs, 1)
            metrics.update(labels, predicted)
        else:
            metrics.update(labels, outputs)
