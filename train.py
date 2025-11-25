from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from rl_core import ActorCritic, SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent to play Snake.")
    parser.add_argument("--total-steps", type=int, default=200_000, help="Total environment steps to run.")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Number of steps per PPO rollout.")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="Optimization epochs per rollout.")
    parser.add_argument("--minibatch-size", type=int, default=128, help="Minibatch size for PPO updates.")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="Clipping epsilon.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden units for policy/value networks.")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"), help="Where to write checkpoints.")
    parser.add_argument("--save-interval", type=float, default=20.0, help="Seconds between checkpoints.")
    parser.add_argument("--log-interval", type=int, default=5_000, help="How often to log progress (steps).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def save_checkpoint(
    path: Path, model: ActorCritic, optimizer: torch.optim.Optimizer, step: int, save_dir: Path
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "timestamp": time.time(),
    }
    torch.save(payload, path)
    torch.save(payload, save_dir / "latest.pt")
    print(f"[checkpoint] saved to {path}")


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = []
    gae = 0.0
    for step in reversed(range(len(rewards))):
        mask = 0.0 if dones[step] else 1.0
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = values[step]
    returns = [a + v for a, v in zip(advantages, values)]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SnakeEnv(seed=args.seed)
    obs = env.reset()
    model = ActorCritic(
        env.observation_size,
        env.action_size,
        hidden_size=args.hidden_size,
        grid_shape=(4, env.grid_height, env.grid_width),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    last_save_time = time.time()
    episode_scores: List[int] = []

    while global_step < args.total_steps:
        vec_obs_buffer = []
        grid_obs_buffer = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []

        for _ in range(args.rollout_steps):
            vec_tensor = torch.tensor(obs["vec"], dtype=torch.float32, device=device).unsqueeze(0)
            grid_tensor = torch.tensor(obs["grid"], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(vec_tensor, grid_tensor)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, done, info = env.step(int(action.item()))
            vec_obs_buffer.append(obs["vec"])
            grid_obs_buffer.append(obs["grid"])
            actions.append(int(action.item()))
            log_probs.append(float(log_prob.cpu().item()))
            values.append(float(value.cpu().item()))
            rewards.append(float(reward))
            dones.append(bool(done))

            obs = env.reset() if done else next_obs
            if done:
                episode_scores.append(info.get("score", 0))

            global_step += 1
            if global_step >= args.total_steps:
                break

        with torch.no_grad():
            next_vec_tensor = torch.tensor(obs["vec"], dtype=torch.float32, device=device).unsqueeze(0)
            next_grid_tensor = torch.tensor(obs["grid"], dtype=torch.float32, device=device).unsqueeze(0)
            _, next_value = model(next_vec_tensor, next_grid_tensor)
            next_value = float(next_value.cpu().item())

        advantages, returns = compute_gae(rewards, values, dones, next_value, args.gamma, args.gae_lambda)

        vec_tensor = torch.tensor(np.array(vec_obs_buffer), dtype=torch.float32, device=device)
        grid_tensor = torch.tensor(np.array(grid_obs_buffer), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.long, device=device)
        log_probs_tensor = torch.tensor(np.array(log_probs), dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)

        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        batch_size = len(vec_obs_buffer)
        for _ in range(args.ppo_epochs):
            idxs = torch.randperm(batch_size)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_idx = idxs[start:end]

                logits, values_pred = model(vec_tensor[mb_idx], grid_tensor[mb_idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_tensor[mb_idx])
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - log_probs_tensor[mb_idx]).exp()
                mb_advantages = advantages_tensor[mb_idx]
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (returns_tensor[mb_idx] - values_pred.squeeze(-1)).pow(2).mean()

                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        if time.time() - last_save_time >= args.save_interval:
            step_ckpt = args.save_dir / f"ppo_snake_{global_step:08d}.pt"
            save_checkpoint(step_ckpt, model, optimizer, global_step, args.save_dir)
            last_save_time = time.time()

        if global_step % args.log_interval < args.rollout_steps:
            mean_score = np.mean(episode_scores[-20:]) if episode_scores else 0.0
            print(
                f"[train] step={global_step} mean_score(last20)={mean_score:.2f} "
                f"episodes={len(episode_scores)}"
            )

    final_ckpt = args.save_dir / f"ppo_snake_{global_step:08d}.pt"
    save_checkpoint(final_ckpt, model, optimizer, global_step, args.save_dir)


if __name__ == "__main__":
    main()
