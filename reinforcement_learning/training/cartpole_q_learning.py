import gym
import numpy as np
import math
from matplotlib import pyplot as plt
from reinforcement_learning.agents import QLearningAgent


def encode(observation: tuple):
    n_buckets = (1, 2, 10, 10)
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    state_bounds[1] = [-0.5, 0.5]
    state_bounds[3] = [-math.radians(50), math.radians(50)]
    state = []

    for i in range(len(observation)):
        if observation[i] <= state_bounds[i][0]:
            bucket_index = 0
        elif observation[i] >= state_bounds[i][1]:
            bucket_index = n_buckets[i] - 1
        else:
            bound_width = state_bounds[i][1] - state_bounds[i][0]
            offset = (n_buckets[i] - 1) * state_bounds[i][0] / bound_width
            scaling = (n_buckets[i] - 1) / bound_width
            bucket_index = int(round(scaling * observation[i] - offset))
        state.append(bucket_index)
    return tuple(state)


env = gym.make("CartPole-v0")
agent = QLearningAgent(env.action_space.n)


def run_episode(max_time=10**4, play=False):
    total_reward = 0.0
    state = encode(env.reset())
    for t in range(max_time):
        if play:
            action = agent.get_best_action(state)
        else:
            action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = encode(next_state)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        if play:
            env.render()
        if done:
            break
    return total_reward


def train():
    rewards = []
    for episode in range(1000):
        rewards.append(run_episode())
        agent.episode_count += 1
        print("EPISODE {}".format(episode))
    print("mean reward {}".format(np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.show()


if __name__ == '__main__':
    train()
    run_episode(play=True)
    env.close()
