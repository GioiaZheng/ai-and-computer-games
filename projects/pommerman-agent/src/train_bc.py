"""Behavioral cloning from Pommerman's built-in SimpleAgent."""

import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import encode_observation
from .environment import make_expert_environment
from .model import ActorCritic, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", default="checkpoints/bc_policy.pt")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def train_batch(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    samples: List[Tuple[np.ndarray, int]],
    device: torch.device,
) -> Tuple[float, float]:
    observations = torch.from_numpy(np.stack([sample[0] for sample in samples])).to(
        device
    )
    actions = torch.tensor([sample[1] for sample in samples], device=device)

    logits, _ = model(observations)
    # Learn every action chosen by the expert. The stricter safety mask belongs
    # to deployment and PPO sampling; applying it here can hide a valid expert
    # label and turn the cross-entropy loss into an artificial 1e9 penalty.
    loss = F.cross_entropy(logits, actions)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    accuracy = (logits.argmax(dim=-1) == actions).float().mean()
    return float(loss.item()), float(accuracy.item())


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    samples: List[Tuple[np.ndarray, int]] = []
    updates = 0
    total_samples = 0
    recent_losses: List[float] = []
    recent_accuracies: List[float] = []

    print("Behavioral cloning from four SimpleAgents")
    print("device={} episodes={}".format(device, args.episodes))
    for episode in range(1, args.episodes + 1):
        environment = make_expert_environment(args.seed + episode)
        observations = environment.reset()
        episode_steps = 0
        done = False
        while not done and episode_steps < args.max_steps:
            actions = environment.act(observations)
            for slot, (observation, action) in enumerate(zip(observations, actions)):
                if 10 + slot not in observation.get("alive", ()):
                    continue
                samples.append(
                    (
                        encode_observation(observation),
                        int(action),
                    )
                )
                total_samples += 1
            observations, _, done, _ = environment.step(actions)
            episode_steps += 1

            while len(samples) >= args.batch_size:
                batch = samples[: args.batch_size]
                del samples[: args.batch_size]
                loss, accuracy = train_batch(model, optimizer, batch, device)
                recent_losses.append(loss)
                recent_accuracies.append(accuracy)
                updates += 1
        environment.close()

        mean_loss = float(np.mean(recent_losses[-20:])) if recent_losses else 0.0
        mean_accuracy = (
            float(np.mean(recent_accuracies[-20:])) if recent_accuracies else 0.0
        )
        print(
            "episode={:04d} steps={:03d} samples={} updates={} loss={:.4f} accuracy={:.3f}".format(
                episode,
                episode_steps,
                total_samples,
                updates,
                mean_loss,
                mean_accuracy,
            )
        )

    if samples:
        loss, accuracy = train_batch(model, optimizer, samples, device)
        recent_losses.append(loss)
        recent_accuracies.append(accuracy)
        updates += 1

    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        {
            "stage": "behavioral_cloning",
            "episodes": args.episodes,
            "samples": total_samples,
            "updates": updates,
            "mean_accuracy": float(np.mean(recent_accuracies[-20:])),
            "seed": args.seed,
        },
    )
    print("saved {}".format(args.checkpoint))


if __name__ == "__main__":
    train(parse_args())
