from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AppSettings:
    gravity: float = 9.81
    wind_x: float = 0.0
    wind_y: float = 0.0
    drag_coefficient: float = 0.05
    restitution: float = 0.8
    fixed_dt: float = 1.0 / 120.0

    def gravity_vector(self) -> np.ndarray:
        return np.array([0.0, -self.gravity], dtype=float)

    def wind_vector(self) -> np.ndarray:
        return np.array([self.wind_x, self.wind_y], dtype=float)
