from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONTACT_EPSILON = 1.0e-9


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
    friction: float
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
        """Advance one Euler step, splitting it at the first box contact."""
        acceleration = self.compute_net_force() / self.ball.mass
        remaining = max(0.0, float(dt))

        while remaining > CONTACT_EPSILON:
            hit_time, hit_axis, hit_normal = self._first_contact(remaining)
            advance = remaining if hit_time is None else hit_time
            self.ball.position = self.ball.position + self.ball.velocity * advance
            self.ball.velocity = self.ball.velocity + acceleration * advance
            remaining -= advance

            if hit_axis is None:
                break

            limit = self.bounds_min[hit_axis] + self.ball.radius if hit_normal > 0 else self.bounds_max[hit_axis] - self.ball.radius
            self.ball.position[hit_axis] = limit
            self._resolve_contact(hit_axis, hit_normal)
            remaining = max(0.0, remaining - CONTACT_EPSILON)

    def _first_contact(self, max_time: float):
        lower = self.bounds_min + self.ball.radius
        upper = self.bounds_max - self.ball.radius
        best_time = None
        best_axis = None
        best_normal = 0

        for axis in range(3):
            velocity = self.ball.velocity[axis]
            if velocity < -1.0e-9 and self.ball.position[axis] <= lower[axis] + CONTACT_EPSILON:
                time = 0.0
                normal = 1
            elif velocity > 1.0e-9 and self.ball.position[axis] >= upper[axis] - CONTACT_EPSILON:
                time = 0.0
                normal = -1
            elif velocity < -1.0e-9 and self.ball.position[axis] > lower[axis]:
                time = (lower[axis] - self.ball.position[axis]) / velocity
                normal = 1
            elif velocity > 1.0e-9 and self.ball.position[axis] < upper[axis]:
                time = (upper[axis] - self.ball.position[axis]) / velocity
                normal = -1
            else:
                continue

            if -CONTACT_EPSILON <= time <= max_time + CONTACT_EPSILON and (best_time is None or time < best_time):
                best_time = max(0.0, time)
                best_axis = axis
                best_normal = normal

        return best_time, best_axis, best_normal

    def _resolve_contact(self, axis: int, normal: int) -> None:
        normal_vector = np.zeros(3, dtype=float)
        normal_vector[axis] = normal
        normal_speed = float(np.dot(self.ball.velocity, normal_vector))
        if normal_speed >= 0.0:
            return

        self.ball.velocity -= (1.0 + self.config.restitution) * normal_speed * normal_vector
        tangent = self.ball.velocity - np.dot(self.ball.velocity, normal_vector) * normal_vector
        self.ball.velocity -= self.config.friction * tangent
