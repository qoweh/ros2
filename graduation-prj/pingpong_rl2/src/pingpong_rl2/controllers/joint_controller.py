from __future__ import annotations

import numpy as np


class JointPositionController:
    def __init__(self, home_targets: np.ndarray) -> None:
        self._home_targets = np.asarray(home_targets, dtype=float).copy()
        self.targets = self._home_targets.copy()

    @property
    def home_targets(self) -> np.ndarray:
        return self._home_targets.copy()

    def reset(self) -> np.ndarray:
        self.targets[:] = self._home_targets
        return self.targets

    def set_targets(self, joint_targets: np.ndarray) -> np.ndarray:
        joint_targets = np.asarray(joint_targets, dtype=float)
        if joint_targets.shape != self._home_targets.shape:
            raise ValueError(
                f"Expected joint target shape {self._home_targets.shape}, got {joint_targets.shape}."
            )
        self.targets[:] = joint_targets
        return self.targets

    def add_joint_offset(self, joint_index: int, offset: float) -> np.ndarray:
        if not 0 <= joint_index < self.targets.size:
            raise IndexError(f"Joint index {joint_index} is out of range for {self.targets.size} joints.")
        self.targets[joint_index] += offset
        return self.targets