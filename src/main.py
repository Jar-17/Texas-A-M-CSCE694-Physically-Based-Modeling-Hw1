from __future__ import annotations

import numpy as np
import pygame

try:
    from physics.world import BallState, PhysicsWorld, SimulationConfig
    from ui.settings import AppSettings
except ModuleNotFoundError:
    from src.physics.world import BallState, PhysicsWorld, SimulationConfig
    from src.ui.settings import AppSettings

WIDTH = 900
HEIGHT = 700
BG = (13, 18, 24)
BALL_COLOR = (255, 195, 80)
BOX_COLOR = (105, 150, 180)
BOX_FILL = (24, 44, 56)
ORIGIN = np.array([0.0, 0.0, 0.0])
SCALE = 210.0


def world_to_screen(pos: np.ndarray) -> tuple[int, int]:
    x = int(WIDTH * 0.5 + (pos[0] - pos[2]) * SCALE * 0.72)
    y = int(HEIGHT * 0.56 - (pos[1] + (pos[0] + pos[2]) * 0.30) * SCALE)
    return x, y


def draw_box(screen: pygame.Surface, bounds_min: np.ndarray, bounds_max: np.ndarray) -> None:
    corners = [
        np.array([x, y, z], dtype=float)
        for y in (bounds_min[1], bounds_max[1])
        for z in (bounds_min[2], bounds_max[2])
        for x in (bounds_min[0], bounds_max[0])
    ]
    points = [world_to_screen(corner) for corner in corners]
    # Corner index is (y, z, x): draw each of the twelve box edges once.
    edges = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 6), (5, 7), (6, 7))
    pygame.draw.polygon(screen, BOX_FILL, [points[i] for i in (0, 1, 3, 2)])
    for first, second in edges:
        pygame.draw.line(screen, BOX_COLOR, points[first], points[second], 2)


def draw_ball(screen: pygame.Surface, ball: BallState) -> None:
    x, y = world_to_screen(ball.position)
    pygame.draw.circle(screen, BALL_COLOR, (x, y), max(2, int(ball.radius * 200)))


def reset_ball(ball: BallState, position, velocity) -> None:
    ball.position = np.array(position, dtype=float)
    ball.velocity = np.array(velocity, dtype=float)


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    settings = AppSettings()
    presets = {
        pygame.K_1: ([-0.8, 0.55, -0.5], [2.8, 1.0, 2.0]),
        pygame.K_2: ([-0.9, -0.2, 0.0], [3.5, 0.4, 0.0]),
        pygame.K_3: ([0.0, -0.7, 0.7], [0.0, 5.0, -3.0]),
        pygame.K_4: ([0.0, 0.0, 0.0], [3.0, 4.5, 2.5]),
    }
    initial_position, initial_velocity = presets[pygame.K_1]
    ball = BallState(position=np.array(initial_position, dtype=float), velocity=np.array(initial_velocity, dtype=float))
    world = PhysicsWorld(
        ball=ball,
        config=SimulationConfig(
            gravity=settings.gravity_vector(),
            wind=settings.wind_vector(),
            drag_coefficient=settings.drag_coefficient,
            restitution=settings.restitution,
            friction=settings.friction,
            time_step=settings.fixed_dt,
        ),
        bounds_min=[-1.5, -1.0, -1.2],
        bounds_max=[1.5, 1.0, 1.2],
    )

    accumulator = 0.0
    running = True
    paused = False
    while running:
        frame_dt = clock.tick(60) / 1000.0
        accumulator += frame_dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    reset_ball(ball, initial_position, initial_velocity)
                elif event.key in presets:
                    initial_position, initial_velocity = presets[event.key]
                    reset_ball(ball, initial_position, initial_velocity)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    settings.wind_x += -0.5 if event.key == pygame.K_LEFT else 0.5
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    settings.fixed_dt = float(np.clip(settings.fixed_dt + (-1 if event.key == pygame.K_UP else 1) / 240.0, 1.0 / 240.0, 1.0 / 20.0))
                elif event.key == pygame.K_f:
                    settings.friction = float(np.clip(settings.friction + 0.05, 0.0, 1.0))
                elif event.key == pygame.K_g:
                    settings.friction = float(np.clip(settings.friction - 0.05, 0.0, 1.0))

        world.config.gravity = settings.gravity_vector()
        world.config.wind = settings.wind_vector()
        world.config.drag_coefficient = settings.drag_coefficient
        world.config.restitution = settings.restitution
        world.config.friction = settings.friction
        world.config.time_step = settings.fixed_dt

        if not paused:
            while accumulator >= settings.fixed_dt:
                world.step(settings.fixed_dt)
                accumulator -= settings.fixed_dt

        screen.fill(BG)
        draw_box(screen, world.bounds_min, world.bounds_max)
        draw_ball(screen, ball)

        info = [
            f"g {settings.gravity:.2f}   wind x {settings.wind_x:+.2f}   drag {settings.drag_coefficient:.3f}",
            f"restitution {settings.restitution:.2f}   friction {settings.friction:.2f}   dt {settings.fixed_dt:.4f}s",
            f"{'PAUSED' if paused else 'RUNNING'}   |   1-4 presets   R reset   SPACE pause   UP/DOWN dt",
            "LEFT/RIGHT wind   F/G friction   ESC quit",
        ]
        for i, line in enumerate(info):
            surf = font.render(line, True, (240, 240, 240))
            screen.blit(surf, (16, 12 + i * 22))

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
