"""Learned controller: scikit-learn classifier for traffic-light control.

Wraps a trained scikit-learn classifier to make safety-guaranteed
lighting decisions by expanding predicted configuration IDs through
the canonical maximal-configuration table.

Usage::

    from learned_controller import LearnedController
    from sklearn_data import load_imitation_model, expand_config_id

    model = load_imitation_model("results/model.pkl")
    controller = LearnedController(model)
    lights = controller.act(sim.state())
"""

import os
import sys
from typing import Any, Optional

import numpy as np

# Ensure scripts path is on sys.path for generate_config
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_HERE, os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from controllers import MAX_PRESSURE_CONFIGS
from generate_config import LANES


def expand_config_id(config_id: int) -> dict[str, bool]:
    """Expand a canonical configuration ID to a full 16-light vector."""
    if config_id < 0 or config_id >= len(MAX_PRESSURE_CONFIGS):
        raise ValueError(f"Configuration ID {config_id} out of range [0, {len(MAX_PRESSURE_CONFIGS) - 1}]")
    return dict(MAX_PRESSURE_CONFIGS[config_id])


class LearnedController:
    """A controller that uses a trained scikit-learn classifier.

    Maintains its own last requested configuration ID, forms the same
    causal feature vector from current state, predicts one ID, updates
    that ID, and expands it through the canonical table to a complete
    16-light vector.
    """

    def __init__(
        self,
        model: Any,
        action_table_version: Optional[str] = None,
    ) -> None:
        if action_table_version is not None and action_table_version != "v1":
            raise ValueError(
                f"Action-table version mismatch: controller expects 'v1', "
                f"got {action_table_version!r}"
            )
        self.model = model
        self.current_config_id = 0  # initial deterministic ID

    def act(self, state: dict) -> dict[str, bool]:
        queues = state.get("queues", {})
        wait_since_served = state.get("wait_since_served", {})

        feature_vec: list[float] = []
        for lane in LANES:
            name = lane.name
            feature_vec.append(float(queues.get(name, 0)))
        for lane in LANES:
            name = lane.name
            feature_vec.append(float(wait_since_served.get(name, 0.0)))
        feature_vec.append(float(self.current_config_id))

        # Predict next configuration ID
        X = np.array([feature_vec])
        if hasattr(self.model, "scaler"):
            X = self.model.scaler.transform(X)  # type: ignore[attr-defined]
        predicted_id = int(self.model.predict(X)[0])

        # Validate predicted ID
        if predicted_id < 0 or predicted_id >= len(MAX_PRESSURE_CONFIGS):
            raise ValueError(
                f"Predicted config ID {predicted_id} out of range [0, {len(MAX_PRESSURE_CONFIGS) - 1}]"
            )

        self.current_config_id = predicted_id
        return expand_config_id(predicted_id)
