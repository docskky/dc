import numpy as np

import torch

from lib import environ


def validation_run(env, net, episodes=100, device="cpu", epsilon=0.02):
    stats = {
        'episode_reward': [],
        'episode_profit': [],
    }

    for episode in range(episodes):
        obs = env.reset()

        total_reward = 0.0
        episode_steps = 0

        while True:
            obs_v = torch.tensor([obs]).to(device)
            out_v = net(obs_v)

            action_idx = out_v.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()

            obs, reward, done, info = env.step(action_idx)

            total_reward += reward

            episode_steps += 1

            if done:
                stats['episode_reward'].append(total_reward)
                stats['episode_profit'].append(info["profit"])
                break

    return { key: np.mean(vals) for key, vals in stats.items() }
