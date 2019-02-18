import numpy as np
import fire
import gym

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value


def compute_returns(next_value, rewards, masks, gamma):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def test_env(env, model, device, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk


def run(cfg=None, lr=1e-3, num_envs=16,
        max_frames=20000, num_steps=5, gamma=.99,
        hidden_size=256, log_freq=1000, use_gpu=True):
    use_gpu = torch.cuda.is_available() and use_gpu
    device ="cuda" if use_gpu else "cpu"

    env_name = "CartPole-v0"
    envs = [make_env(env_name) for i in range(num_envs)]
    envs = SubprocVecEnv(envs)
    env = gym.make(env_name)

    num_inputs  = envs.observation_space.shape[0]
    num_outputs = envs.action_space.n

    model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    frame_idx    = 0
    test_rewards = []

    state = envs.reset()

    while frame_idx < max_frames:

        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

            state = next_state
            frame_idx += 1

            if frame_idx % log_freq == 0:
                test_rewards.append(np.mean([test_env(env, model, device) for _ in range(10)]))
                print(f"Frame {frame_idx:6d} len {len(test_rewards):4d} reward {np.mean(test_rewards[-10:]):6.2f}")

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks, gamma)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    fire.Fire(run)

