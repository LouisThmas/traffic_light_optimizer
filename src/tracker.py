"""Lightweight TensorBoard scalar tracker.

Creates standard TensorBoard event files that can be read by
``tensorboard --logdir <dir>`` for visualization of scalar metrics
(training loss, evaluation scores, etc.) over time.

Usage::

    from tensorboard.summary import Writer
    from experiment import ExperimentTracker

    writer = Writer("/path/to/logs")
    tracker = ExperimentTracker(writer)
    tracker.log_scalar("training/loss", 0.5, step=42)
    tracker.close()

Then run: ``tensorboard --logdir /path/to/logs``
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorboard.summary import Writer as _Writer


class ExperimentTracker:
    """Write scalar values to TensorBoard event files.

    Accepts ``(tag, value, step)`` calls so later training modules can
    log without changing the artifact layout.
    """

    def __init__(self, writer: "_Writer") -> None:
        self._writer = writer

    def log_scalar(self, tag: str, value: float, step: int = 0) -> None:
        """Log a single scalar value at a given step."""
        self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        """Flush and close the writer."""
        self._writer.close()
