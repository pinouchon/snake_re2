from __future__ import annotations

import random
import sys
from collections import deque

import pygame


CELL_SIZE = 20
GRID_WIDTH = 8
GRID_HEIGHT = 8
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
SNAKE_INITIAL_LENGTH = 3
MOVE_DELAY_MS = 120
FONT_NAME = "arial"


DIRECTIONS = {
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0),
}


def draw_block(surface: pygame.Surface, color: pygame.Color, position: tuple[int, int]) -> None:
    rect = pygame.Rect(position[0] * CELL_SIZE, position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, color, rect)


def cell_center(position: tuple[int, int]) -> tuple[int, int]:
    x = position[0] * CELL_SIZE + CELL_SIZE // 2
    y = position[1] * CELL_SIZE + CELL_SIZE // 2
    return x, y


def draw_snake(surface: pygame.Surface, snake: deque[tuple[int, int]]) -> None:
    head_color = pygame.Color("lime")
    body_color = pygame.Color("green3")
    if len(snake) > 1:
        width = max(1, CELL_SIZE - 6)
        for i in range(len(snake) - 1):
            pygame.draw.line(surface, body_color, cell_center(snake[i]), cell_center(snake[i + 1]), width)
    for idx, part in enumerate(snake):
        color = head_color if idx == 0 else body_color
        radius = max(2, CELL_SIZE // 2 - 2)
        pygame.draw.circle(surface, color, cell_center(part), radius)


def random_empty_cell(exclude: set[tuple[int, int]]) -> tuple[int, int]:
    while True:
        cell = (random.randrange(GRID_WIDTH), random.randrange(GRID_HEIGHT))
        if cell not in exclude:
            return cell


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(FONT_NAME, 24)

    snake = deque([(GRID_WIDTH // 2 - i, GRID_HEIGHT // 2) for i in range(SNAKE_INITIAL_LENGTH)])
    direction = (1, 0)
    pending_direction = direction
    food = random_empty_cell(set(snake))
    score = 0
    move_event = pygame.USEREVENT + 1
    pygame.time.set_timer(move_event, MOVE_DELAY_MS)
    alive = True

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    sys.exit()
                if event.key in DIRECTIONS:
                    new_dir = DIRECTIONS[event.key]
                    if (new_dir[0] != -direction[0]) or (new_dir[1] != -direction[1]):
                        pending_direction = new_dir
                if not alive and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    snake = deque([(GRID_WIDTH // 2 - i, GRID_HEIGHT // 2) for i in range(SNAKE_INITIAL_LENGTH)])
                    direction = (1, 0)
                    pending_direction = direction
                    food = random_empty_cell(set(snake))
                    score = 0
                    alive = True
            if event.type == move_event and alive:
                direction = pending_direction
                head_x, head_y = snake[0]
                dx, dy = direction
                new_head = (head_x + dx, head_y + dy)

                # Wall collision ends the round.
                if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
                    alive = False
                elif new_head in snake:
                    alive = False
                else:
                    snake.appendleft(new_head)
                    if new_head == food:
                        score += 1
                        food = random_empty_cell(set(snake))
                    else:
                        snake.pop()

        screen.fill((24, 24, 24))
        draw_block(screen, pygame.Color("red"), food)

        draw_snake(screen, snake)

        score_text = font.render(f"Score: {score}", True, pygame.Color("white"))
        screen.blit(score_text, (10, 10))

        if not alive:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            screen.blit(overlay, (0, 0))
            msg = font.render("Game Over - press Space to restart or Esc to quit", True, pygame.Color("white"))
            rect = msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(msg, rect)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
