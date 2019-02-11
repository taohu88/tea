from itertools import count
import numpy as np
import gym
import fire

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from tea.config.app_cfg import AppConfig
import tea.models.factory as MFactory


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


def select_action(policy, state):
    state = torch.from_numpy(state[np.newaxis,:]).float()
    probs, state_value = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action), state_value


def train_after_episode(actions_, rewards_, optimizer, gamma, eps):
    R = 0
    saved_actions = actions_
    policy_losses = []
    value_losses = []
    rewards = []
    for r in rewards_[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()


def create_env():
    return gym.make('CartPole-v0')

"""
It is good to follow pattern.
In this case, any application starts with cfg file, 
with optional override arguments like the following: 
    model_cfg
    model_out_dir
    epochs, lr, batch etc
"""
def run(ini_file='rl-actor-critic.ini',
        model_cfg='./rl-cartpolo-actor-critic.cfg',
        model_out_dir='./models',
        gamma=0.99,
        lr=1e-2,
        log_freq=10,
        render=True,
        use_gpu=True,
        explore_lr=False):
    # Step 1: parse config
    cfg = AppConfig.from_file(ini_file,
                                model_cfg=model_cfg,
                                model_out_dir=model_out_dir,
                                gamma=gamma,
                                lr=lr,
                                log_freq=log_freq,
                                render=render,
                                use_gpu=use_gpu)
    cfg.print()
    # Step 2: create env
    env = create_env()
    print(f"Env is {env}")

    # Step 3: create model
    policy = MFactory.create_model(cfg)
    print(policy)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    eps = np.finfo(np.float32).eps.item()

    reward_threshold = env.spec.reward_threshold
    running_reward = 10
    max_run_len = 10000
    print(f"Reward threshold {reward_threshold} gamma is {gamma} eps {eps}")

    for i_episode in count(1):
        state = env.reset()
        rewards = []
        actions = []
        # run one episode
        for run_len in range(max_run_len):  # Don't infinite loop while learning
            action, log_prob, state_value = select_action(policy, state)
            state, reward, done, _ = env.step(action)
            if render:
                env.render()
            rewards.append(reward)
            actions.append((log_prob, state_value))
            if done:
                break

        # train after one episode
        running_reward = running_reward * 0.99 + run_len * 0.01
        train_after_episode(actions, rewards, optimizer, gamma, eps)

        if i_episode % log_freq == 0:
            print('Episode {}\tLast length: {:4d}\tAverage length: {:.2f}'.format(
                i_episode, run_len, running_reward))
        if running_reward > reward_threshold:
            print("Solved! Running reward is now {:.3f} and "
                  "the last episode runs to {:4d} time steps!".format(running_reward, run_len))
            break


if __name__ == '__main__':
    fire.Fire(run)
