from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
from torch import nn


GRID_WIDTH = 8
GRID_HEIGHT = 8
SNAKE_INITIAL_LENGTH = 3
FOOD_REWARD = 1.0
STEP_REWARD = -0.01
COLLISION_REWARD = -1.0

# Clockwise order makes turning easy: up, right, down, left.
DIR_VECTORS: Tuple[Tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


class RecurrentActorCritic(nn.Module):
    """GRU policy/value network (kept for loading older checkpoints)."""

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(obs_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        elif obs.dim() != 3:
            raise ValueError("obs must have shape (batch, obs) or (batch, seq, obs)")

        x = torch.relu(self.encoder(obs))
        rnn_out, new_hidden = self.gru(x, hidden)
        last_out = rnn_out[:, -1]
        logits = self.policy_head(last_out)
        value = self.value_head(last_out).squeeze(-1)
        return logits, value, new_hidden

    def act(self, obs: np.ndarray | torch.Tensor, hidden: torch.Tensor) -> tuple[int, float, float, torch.Tensor]:
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        else:
            obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs
        logits, value, new_hidden = self.forward(obs_tensor, hidden)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item()), new_hidden.detach()


class SnakeEnv:
    """Simple Snake environment for RL training (no rendering)."""

    def __init__(self, grid_width: int = GRID_WIDTH, grid_height: int = GRID_HEIGHT, seed: int | None = None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.rng = random.Random(seed)
        self.direction_idx = 1  # start moving right
        self.snake: Deque[Tuple[int, int]] = deque()
        self.food: tuple[int, int] = (0, 0)
        self.score = 0
        self.done = False
        self._obs_size = 7  # danger straight/left/right + food ahead/left/right/behind
        self.reset()

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        # Actions are relative to current heading: 0=straight, 1=turn left, 2=turn right.
        return 3

    @property
    def direction(self) -> tuple[int, int]:
        return DIR_VECTORS[self.direction_idx]

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
        center_x = self.grid_width // 2
        center_y = self.grid_height // 2
        self.direction_idx = 1  # right
        self.snake = deque([(center_x - i, center_y) for i in range(SNAKE_INITIAL_LENGTH)])
        self.food = self._spawn_food()
        self.score = 0
        self.done = False
        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if self.done:
            return self._get_obs(), 0.0, True, {"score": self.score}

        self._apply_action(action)
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        reward = STEP_REWARD
        if self._would_collide(new_head):
            self.done = True
            reward = COLLISION_REWARD
            return self._get_obs(), reward, True, {"score": self.score}

        self.snake.appendleft(new_head)
        if new_head == self.food:
            self.score += 1
            reward = FOOD_REWARD
            self.food = self._spawn_food()
        else:
            self.snake.pop()

        return self._get_obs(), reward, False, {"score": self.score}

    # Internal helpers.
    def _apply_action(self, action: int) -> None:
        if action == 1:  # turn left
            self.direction_idx = (self.direction_idx - 1) % 4
        elif action == 2:  # turn right
            self.direction_idx = (self.direction_idx + 1) % 4
        # action == 0 means keep heading

    def _would_collide(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        return pos in self.snake

    def _spawn_food(self) -> tuple[int, int]:
        occupied = set(self.snake)
        while True:
            cell = (self.rng.randrange(self.grid_width), self.rng.randrange(self.grid_height))
            if cell not in occupied:
                return cell

    def _danger_flags(self) -> tuple[float, float, float]:
        forward = self.direction
        left = (-forward[1], forward[0])
        right = (forward[1], -forward[0])
        head_x, head_y = self.snake[0]
        forward_pos = (head_x + forward[0], head_y + forward[1])
        left_pos = (head_x + left[0], head_y + left[1])
        right_pos = (head_x + right[0], head_y + right[1])
        return (
            float(self._would_collide(forward_pos)),
            float(self._would_collide(left_pos)),
            float(self._would_collide(right_pos)),
        )

    def _food_direction_flags(self) -> tuple[float, float, float, float]:
        head_x, head_y = self.snake[0]
        food_dx = self.food[0] - head_x
        food_dy = self.food[1] - head_y
        forward = self.direction
        left = (-forward[1], forward[0])
        right = (forward[1], -forward[0])
        backward = (-forward[0], -forward[1])

        def dot(a: tuple[int, int], b: tuple[int, int]) -> int:
            return a[0] * b[0] + a[1] * b[1]

        delta = (food_dx, food_dy)
        return (
            float(dot(delta, forward) > 0),
            float(dot(delta, left) > 0),
            float(dot(delta, right) > 0),
            float(dot(delta, backward) > 0),
        )

    def _get_obs(self) -> np.ndarray:
        danger = self._danger_flags()
        food_dirs = self._food_direction_flags()
        obs = np.array(
            [
                danger[0],
                danger[1],
                danger[2],
                food_dirs[0],
                food_dirs[1],
                food_dirs[2],
                food_dirs[3],
            ],
            dtype=np.float32,
        )
        grid = self._encode_grid()
        return {"vec": obs, "grid": grid}

    def _encode_grid(self) -> np.ndarray:
        # One-hot channels: 0=head, 1=body (excluding head), 2=food, 3=empty
        grid = np.zeros((4, self.grid_height, self.grid_width), dtype=np.float32)
        # start with empty = 1 everywhere
        grid[3, :, :] = 1.0

        food_x, food_y = self.food
        grid[2, food_y, food_x] = 1.0
        grid[3, food_y, food_x] = 0.0

        snake_list = list(self.snake)
        if snake_list:
            head_x, head_y = snake_list[0]
            grid[0, head_y, head_x] = 1.0
            grid[3, head_y, head_x] = 0.0
            for part in snake_list[1:]:
                grid[1, part[1], part[0]] = 1.0
                grid[3, part[1], part[0]] = 0.0
        return grid


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_size: int = 128,
        cnn_channels: int = 32,
        grid_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.grid_shape = grid_shape or (4, GRID_HEIGHT, GRID_WIDTH)
        c, h, w = self.grid_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *self.grid_shape)
            cnn_out = self.cnn(dummy)
            self.cnn_out_dim = cnn_out.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        fusion_dim = hidden_size + self.cnn_out_dim
        self.policy_head = nn.Linear(fusion_dim, action_size)
        self.value_head = nn.Linear(fusion_dim, 1)

    def forward(self, vec_obs: torch.Tensor, grid_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mlp_out = self.mlp(vec_obs)
        cnn_out = self.cnn(grid_obs)
        fused = torch.cat([mlp_out, cnn_out], dim=-1)
        logits = self.policy_head(fused)
        value = self.value_head(fused).squeeze(-1)
        return logits, value

    def act(
        self, obs: dict[str, np.ndarray] | dict[str, torch.Tensor]
    ) -> tuple[int, float, float]:
        vec = obs["vec"]
        grid = obs["grid"]
        if isinstance(vec, np.ndarray):
            vec_tensor = torch.from_numpy(vec).float().unsqueeze(0)
        else:
            vec_tensor = vec.unsqueeze(0) if vec.dim() == 1 else vec
        if isinstance(grid, np.ndarray):
            grid_tensor = torch.from_numpy(grid).float().unsqueeze(0)
        else:
            grid_tensor = grid.unsqueeze(0) if grid.dim() == 3 else grid
        logits, value = self.forward(vec_tensor, grid_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())
