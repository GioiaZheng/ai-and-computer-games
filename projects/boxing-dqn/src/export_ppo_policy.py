"""Export an SB3 PPO Atari policy as a small teacher-compatible state dict."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Inference-only network matching SB3's NatureCNN and policy head."""

    def __init__(self, in_channels=6, num_actions=18):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, observation):
        observation = observation.permute(0, 3, 1, 2).float() / 255.0
        features = self.conv(observation).reshape(observation.shape[0], -1)
        return self.fc(features)


def export_policy(model_path: Path, output_path: Path):
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    source = model.policy
    target = QNetwork()
    target.conv.load_state_dict(source.features_extractor.cnn[:6].state_dict())
    target.fc[0].load_state_dict(source.features_extractor.linear[0].state_dict())
    target.fc[2].load_state_dict(source.action_net.state_dict())

    generator = torch.Generator().manual_seed(0)
    sample = torch.randint(
        0, 256, (2, 84, 84, 6), dtype=torch.uint8, generator=generator
    )
    with torch.inference_mode():
        exported_logits = target(sample)
        features = source.extract_features(sample.permute(0, 3, 1, 2))
        latent_policy, _ = source.mlp_extractor(features)
        source_logits = source.action_net(latent_policy)
    if not torch.allclose(exported_logits, source_logits, atol=1e-6):
        raise RuntimeError("Exported policy logits do not match the SB3 policy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target.state_dict(), output_path)
    print(f"Exported PPO policy weights: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_policy(args.model, args.output)
