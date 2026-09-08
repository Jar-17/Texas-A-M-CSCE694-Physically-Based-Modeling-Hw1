from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BallState:
    position: np.ndarray
    velocity: np.ndarray
    mass: float = 1.0
    radius: float = 0.06


@dataclass
class SimulationConfig:
    gravity: np.ndarray
    wind: np.ndarray
    drag_coefficient: float
    restitution: float
    time_step: float


class PhysicsWorld:
    def __init__(self, ball: BallState, config: SimulationConfig, bounds_min, bounds_max):
        self.ball = ball
        self.config = config
        self.bounds_min = np.array(bounds_min, dtype=float)
        self.bounds_max = np.array(bounds_max, dtype=float)

    def compute_net_force(self) -> np.ndarray:
        gravity_force = self.ball.mass * self.config.gravity
        wind_force = np.array(self.config.wind, dtype=float)
        drag_force = -self.config.drag_coefficient * self.ball.velocity
        return gravity_force + wind_force + drag_force

    def step(self, dt: float) -> None:
        acceleration = self.compute_net_force() / self.ball.mass
        self.ball.position = self.ball.position + self.ball.velocity * dt
        self.ball.velocity = self.ball.velocity + acceleration * dt
