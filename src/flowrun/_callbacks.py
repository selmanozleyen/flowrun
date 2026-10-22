"""Training callbacks that are flowrun's own."""

from __future__ import annotations

from lightning.pytorch import Callback, LightningModule, Trainer


class BestMetrics(Callback):
    """Every val / test metric of the epoch where the monitored metric was best, for the run's summary.

    Test metrics are read at the best validation epoch, not at their own best, so they stay unseen by selection.
    """

    def __init__(self, monitor: str, mode: str = "min") -> None:
        self.monitor, self.mode = monitor, mode
        self.best: float | None = None
        self.metrics: dict[str, float] = {}
        self.epoch: int | None = None

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        value = trainer.callback_metrics.get(self.monitor)
        if trainer.sanity_checking or value is None:
            return
        value = float(value)  # one host read per validation
        if self.best is None or (value < self.best if self.mode == "min" else value > self.best):
            self.best, self.epoch = value, trainer.current_epoch
            self.metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith(("val/", "test/"))}

    def summary(self) -> dict[str, float | int]:
        """``best/<metric>`` and ``best/epoch``; empty if the monitored metric was never logged."""
        if self.epoch is None:
            return {}
        return {**{f"best/{k}": v for k, v in self.metrics.items()}, "best/epoch": self.epoch}
