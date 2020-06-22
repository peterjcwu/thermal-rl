import numpy as np
from matplotlib import pyplot as plt
from q_learning_agent import QLearningAgent
from heater_env import HeaterEnv

env = HeaterEnv()
agent = QLearningAgent(env.action_space.n)


def encode(observation: tuple):
    n_buckets = (3, 11)
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
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


def run_episode(max_time=100):
    total_reward = 0.0
    state = encode(env.reset())
    for t in range(max_time):
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = encode(next_state)
        agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        env.render()
        if done:
            break
    return total_reward


def train():
    rewards = []
    tcs = []
    tjs = []
    heater_powers = []
    for episode in range(100):
        rewards.append(run_episode())
        tcs.append(env.tc_value)
        tjs.append(env.tj_value)
        heater_powers.append(env.heater_power)
        agent.episode_count += 1
        print("EPISODE {}".format(episode))
    print("mean reward {}".format(np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.plot(tcs)
    plt.plot(tjs)
    plt.plot(heater_powers)
    plt.show()


if __name__ == '__main__':
    train()
