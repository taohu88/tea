from itertools import count
import numpy as np
import gym
import fire

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from tea.config.app_cfg import AppConfig
import tea.models.factory as MFactory
from tea.utils.commons import islist, discouont_rewards


class OnPolicyLearner():

    def __init__(self, cfg, env, policy, lr):
        self.cfg = cfg
        self.env = env
        self.policy = policy
        # TODO fix create only adam optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)


    @staticmethod
    def select_action(policy, state):
        if isinstance(state, np.ndarray):
            if len(state.shape) < 2:
                # make it look like bactch
                state = state[np.newaxis, :]
            state = torch.from_numpy(state).float()
        elif isinstance(state, torch.tensor):
            if len(state.size()) < 2:
                state = state[None,...]
        else:
            raise Exception(f"Unexcepted type of {type(state)}")

        output = policy(state)
        if islist(output):
            probs = output[0]
        else:
            probs = output
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), output[1:]

    @staticmethod
    def train_after_episode(policy_outs, raw_rewards, optimizer, gamma, eps):
        policy_losses = []
        value_losses = []

        rewards = torch.tensor(discouont_rewards(raw_rewards, gamma))
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for (log_prob, value), r in zip(policy_outs, rewards):
            if len(value) > 0:
                value = value[0]
                reward = r - value.item()
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
            policy_losses.append(-log_prob * reward)
        optimizer.zero_grad()
        if value_losses:
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        else:
            loss = torch.stack(policy_losses).sum()
        loss.backward()
        optimizer.step()

    # This is like stream online corresponding batch in other use cases
    def run_one_episode(self, policy, max_run_per_episode):
        state = self.env.reset()
        rewards = []
        policy_outs = []
        # run one episode
        for run_len in range(max_run_per_episode):
            action, log_prob, state_value = self.select_action(policy, state)
            state, reward, done, _ = self.env.step(action)
            # if self.render:
            #     self.env.render()
            rewards.append(reward)
            if len(state_value) > 0:
                policy_outs.append((log_prob, state_value))
            else:
                policy_outs.append(log_prob)
            if done:
                break
        return policy_outs, rewards, run_len

    def fit(self, lr, max_episodes, gamma=0.99, reward_threshold=195, max_run_per_episode=10000):
        log_freq = self.cfg.get_log_freq()
        device = self.cfg.get_device()
        policy = self.policy.to(device)
        eps = self.cfg.get_eps()

        running_reward = reward_threshold // 20
        for i_episode in range(max_episodes):
            actions, rewards, run_len = self.run_one_episode(policy, max_run_per_episode)
            # train after one episode
            running_reward = running_reward * 0.99 + run_len * 0.01
            self.train_after_episode(actions, rewards, self.optimizer, gamma, eps)

            if i_episode % log_freq == 0:
                print('Episode {}\tLast length: {:4d}\tAverage length: {:.2f}'.format(
                    i_episode, run_len, running_reward))
            if running_reward > reward_threshold:
                print("Solved! Running reward is now {:.3f} and "
                      "the last episode runs to {:4d} time steps!".format(running_reward, run_len))
                break


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
        render=False,
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
    # Step 2: create env
    env = create_env()
    print(f"Env is {env}")

    eps = np.finfo(np.float32).eps.item()
    cfg.update(eps=eps)
    cfg.print()

    # Step 3: create model
    policy = MFactory.create_model(cfg)
    print(policy)

    max_episodes = 10000
    reward_threshold = env.spec.reward_threshold
    max_run_per_episode = 10000
    print(f"Reward threshold {reward_threshold} gamma is {gamma}")

    # Step 4: create learner
    learner = OnPolicyLearner(cfg, env, policy, lr)
    learner.fit(lr, max_episodes=max_episodes,
                    gamma=gamma,
                    reward_threshold=reward_threshold,
                    max_run_per_episode=max_run_per_episode)


if __name__ == '__main__':
    fire.Fire(run)
