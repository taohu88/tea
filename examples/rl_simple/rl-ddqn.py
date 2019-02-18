import math
import gym
import fire
import torch
import torch.optim as optim
import numpy as np
from rl.replayer_buffer import ReplayBuffer
from rl.models import CartPoloDQN


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def train_batch(current_model, target_model, batches, optimizer, gamma=0.99, device=None):
    state, action, reward, next_state, done = batches

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values = current_model(state)
    with torch.no_grad():
        next_q_values = current_model(next_state).detach()
    with torch.no_grad():
        next_q_state_values = target_model(next_state).detach()

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_best_action = torch.max(next_q_values, 1)[1]
    next_q_value = next_q_state_values.gather(1, next_best_action.unsqueeze(1)).squeeze(1)
    # If we don't squeeze
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = (q_value - expected_q_value).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def run(cfg=None, lr=1e-3, max_runs=10000, gamma=.99, replay_sz=5000, batch_size=64, log_freq=200, use_gpu=True):
    use_gpu = torch.cuda.is_available() and use_gpu
    device ="cuda" if use_gpu else "cpu"

    env_id = "CartPole-v0"
    env = gym.make(env_id)

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    epsilon_by_frame = lambda frame_idx: epsilon_final + \
                        (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    current_model = CartPoloDQN(env.observation_space.shape[0], env.action_space.n)
    target_model = CartPoloDQN(env.observation_space.shape[0], env.action_space.n)
    current_model = current_model.to(device)
    target_model = target_model.to(device)
    update_target(current_model, target_model)

    optimizer = optim.Adam(current_model.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(replay_sz)

    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()
    for frame_idx in range(1, max_runs + 1):
        epsilon = epsilon_by_frame(frame_idx)
        action = current_model.act(state, epsilon, device)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0

        if len(replay_buffer) > batch_size:
            batches = replay_buffer.sample(batch_size)
            loss = train_batch(current_model, target_model, batches, optimizer, gamma, device)
            losses.append(loss.item())

        if frame_idx % log_freq == 0:
            print(f"Frame {frame_idx:6d} len {len(all_rewards):4d} reward {np.mean(all_rewards[-10:]):6.2f} loss {np.mean(losses[-200:]):8.3f}")

        if frame_idx % 100 == 0:
            update_target(current_model, target_model)


if __name__ == '__main__':
    fire.Fire(run)

