from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pygame
import torch

from rl_core import ActorCritic, SnakeEnv
from snake import CELL_SIZE, GRID_HEIGHT, GRID_WIDTH, MOVE_DELAY_MS, SCREEN_HEIGHT, SCREEN_WIDTH, draw_block, draw_snake


def find_checkpoint(path: Optional[str]) -> Path:
    if path:
        return Path(path)
    latest = Path("checkpoints/latest.pt")
    if latest.exists():
        return latest
    ckpt_dir = Path("checkpoints")
    if ckpt_dir.exists():
        candidates = sorted(ckpt_dir.glob("ppo_snake_*.pt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            return candidates[-1]
    raise FileNotFoundError("No checkpoint found. Provide --checkpoint or run training first.")


def load_policy(checkpoint_path: Path, env: SnakeEnv, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload["model_state"]
    model = ActorCritic(
        env.observation_size, env.action_size, grid_shape=(4, env.grid_height, env.grid_width)
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def select_action(model: ActorCritic, obs, device: torch.device) -> int:
    vec_tensor = torch.tensor(obs["vec"], dtype=torch.float32, device=device).unsqueeze(0)
    grid_tensor = torch.tensor(obs["grid"], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(vec_tensor, grid_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
    return int(action.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a trained Snake policy.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (defaults to latest).")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake - Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

    env = SnakeEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = find_checkpoint(args.checkpoint)
    model = load_policy(ckpt_path, env, device)

    obs = env.reset()
    agent_speed_ms = max(1, MOVE_DELAY_MS // 5)
    move_event = pygame.USEREVENT + 2
    pygame.time.set_timer(move_event, agent_speed_ms)
    paused_until_ms: int | None = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                pygame.quit()
                return
            if event.type == move_event:
                now_ms = pygame.time.get_ticks()
                if paused_until_ms is not None:
                    if now_ms < paused_until_ms:
                        continue
                    paused_until_ms = None
                    obs = env.reset()
                    continue
                action = select_action(model, obs, device)
                obs, reward, done, info = env.step(action)
                if done:
                    paused_until_ms = now_ms + 1000  # pause for 1 second before restarting

        screen.fill((24, 24, 24))
        draw_block(screen, pygame.Color("red"), env.food)
        draw_snake(screen, env.snake)

        score_text = font.render(f"Score: {env.score}", True, pygame.Color("white"))
        ckpt_text = font.render(f"Checkpoint: {ckpt_path.name}", True, pygame.Color("gray70"))
        screen.blit(score_text, (10, 10))
        screen.blit(ckpt_text, (10, 36))

        if paused_until_ms is not None:
            msg = font.render("Collision - restarting...", True, pygame.Color("white"))
            msg_rect = msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(msg, msg_rect)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
