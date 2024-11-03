import wandb

from utils.metrics import Metrics


class WBLogger:
    def __init__(self, train_metrics: Metrics, val_metrics: Metrics):
        self._train_metrics = train_metrics
        self._val_metrics = val_metrics

    def log_train_epoch(self, num: int, loss: float, **kwargs):
        logs = {
            "Epoch": num,
            "Train Loss": loss,
            **kwargs,
        }
        logs.update(self._train_metrics.to_dict())
        wandb.log(logs)

    def log_val_epoch(self, num: int, loss: float, **kwargs):
        logs = {
            "Epoch": num,
            "Valid Loss": loss,
            **kwargs,
        }
        logs.update(self._val_metrics.to_dict())
        wandb.log(logs)
