from __future__ import annotations

import numpy as np
import pygame

from physics.world import BallState, PhysicsWorld, SimulationConfig
from ui.settings import AppSettings

WIDTH = 900
HEIGHT = 700
BG = (20, 20, 28)
BALL_COLOR = (255, 195, 80)
BOX_COLOR = (180, 180, 200)


def world_to_screen(pos: np.ndarray) -> tuple[int, int]:
    x = int(WIDTH * 0.5 + pos[0] * 200)
    y = int(HEIGHT * 0.5 - pos[1] * 200)
    return x, y


def draw_box(screen: pygame.Surface) -> None:
    margin = 80
    pygame.draw.rect(screen, BOX_COLOR, pygame.Rect(margin, margin, WIDTH - 2 * margin, HEIGHT - 2 * margin), 3)


def draw_ball(screen: pygame.Surface, ball: BallState) -> None:
    x, y = world_to_screen(ball.position)
    pygame.draw.circle(screen, BALL_COLOR, (x, y), max(2, int(ball.radius * 200)))


def main() -> int:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    settings = AppSettings()
    ball = BallState(position=np.array([0.0, 0.6], dtype=float), velocity=np.array([0.8, 0.0], dtype=float))
    world = PhysicsWorld(
        ball=ball,
        config=SimulationConfig(
            gravity=settings.gravity_vector(),
            wind=settings.wind_vector(),
            drag_coefficient=settings.drag_coefficient,
            restitution=settings.restitution,
            time_step=settings.fixed_dt,
        ),
        bounds_min=[-1.5, -1.0],
        bounds_max=[1.5, 1.0],
    )

    accumulator = 0.0
    running = True
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
                    ball.position = np.array([0.0, 0.6], dtype=float)
                    ball.velocity = np.array([0.8, 0.0], dtype=float)

        world.config.gravity = settings.gravity_vector()
        world.config.wind = settings.wind_vector()
        world.config.drag_coefficient = settings.drag_coefficient
        world.config.restitution = settings.restitution
        world.config.time_step = settings.fixed_dt

        while accumulator >= settings.fixed_dt:
            world.step(settings.fixed_dt)
            accumulator -= settings.fixed_dt

        screen.fill(BG)
        draw_box(screen)
        draw_ball(screen, ball)

        info = [
            f"Gravity: {settings.gravity:.2f}   Wind: ({settings.wind_x:.2f}, {settings.wind_y:.2f})",
            f"Drag: {settings.drag_coefficient:.3f}   Restitution: {settings.restitution:.2f}",
            "R = reset   ESC = quit",
        ]
        for i, line in enumerate(info):
            surf = font.render(line, True, (240, 240, 240))
            screen.blit(surf, (16, 12 + i * 22))

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
