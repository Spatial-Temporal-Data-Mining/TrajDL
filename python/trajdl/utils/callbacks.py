# Copyright 2024 All authors of TrajDL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable PyTorch Lightning callbacks for TrajDL training scripts.

These callbacks are designed to be model-agnostic and depend only on
public Lightning APIs (``trainer.callback_metrics``, ``trainer.current_epoch``,
``trainer.global_step``), so they can be plugged into any benchmark's
``L.Trainer(callbacks=[...])`` without modification.
"""

from lightning.pytorch.callbacks import Callback


class EpochLogPrinter(Callback):
    """Print epoch-level metrics to stdout for headless environments.

    Lightning's default ``TQDMProgressBar`` auto-disables itself in non-TTY
    contexts (e.g. cloud batch-job container logs, ``kubectl logs``, redirected stdout),
    leaving the user without any per-epoch feedback. This callback prints a
    single line at the end of each train and validation epoch summarizing
    every logged metric, so training progress is visible in plain log files.

    The printer reads from ``trainer.callback_metrics`` (the same dict that
    feeds Lightning's loggers and ``ModelCheckpoint``), so any metric that
    was registered via ``self.log(...)`` in the LightningModule will appear
    automatically — no need to enumerate metric names per algorithm.

    Output format::

        [epoch 0 train-end] step=40000 loss=0.123 | acc=0.95 | ...
        [epoch 0 val-end]   step=40000 val/loss=0.111 | val/acc=0.96

    Notes
    -----
    - Sanity-check epochs are skipped to avoid noisy output before training.
    - When used together with progress bars, prefer
      ``L.Trainer(enable_progress_bar=False, callbacks=[..., EpochLogPrinter()])``
      to avoid mixing carriage-return tqdm output with line-based prints in
      log viewers that cannot render ``\\r``.

    Examples
    --------
    >>> from trajdl.utils import EpochLogPrinter
    >>> trainer = L.Trainer(
    ...     max_epochs=10,
    ...     callbacks=[EpochLogPrinter()],
    ...     enable_progress_bar=False,
    ... )
    """

    def on_train_epoch_end(self, trainer, pl_module):
        self._print_metrics(trainer, tag="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._print_metrics(trainer, tag="val")

    @staticmethod
    def _print_metrics(trainer, tag):
        if trainer.sanity_checking:
            return
        metrics = {
            k: (v.item() if hasattr(v, "item") else float(v))
            for k, v in trainer.callback_metrics.items()
        }
        if not metrics:
            return
        formatted = " | ".join(f"{k}={v:.6f}" for k, v in sorted(metrics.items()))
        print(
            f"[epoch {trainer.current_epoch} {tag}-end] "
            f"step={trainer.global_step} {formatted}",
            flush=True,
        )
