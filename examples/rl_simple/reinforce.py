import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def select_action(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(policy, optimizer, gamma, eps):
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='G',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-freq', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    # env.seed(args.seed)
    # torch.manual_seed(args.seed)

    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    eps = np.finfo(np.float32).eps.item()

    gamma = args.gamma
    log_freq = args.log_freq
    reward_threshold = env.spec.reward_threshold
    running_reward = 10
    max_run_len = 10000
    print(f"{env} specific reward threshold {reward_threshold} gamma is {gamma} eps {eps}")

    for i_episode in count(1):
        state = env.reset()
        for run_len in range(max_run_len):  # Don't infinite loop while learning
            action = select_action(policy, state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + run_len * 0.01
        finish_episode(policy, optimizer, gamma, eps)
        if i_episode % log_freq == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, run_len, running_reward))
        if running_reward > reward_threshold:
            print("Solved! Running reward is now {:.3f} and "
                  "the last episode runs to {:5d} time steps!".format(running_reward, run_len))
            break


if __name__ == '__main__':
    main()
