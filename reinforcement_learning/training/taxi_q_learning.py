import gym
import numpy as np
from matplotlib import pyplot as plt
from q_learning_agent import QLearningAgent

env = gym.make("Taxi-v2")
agent = QLearningAgent(env.action_space.n)


def run_episode(max_time=10**4, play=False):
    total_reward = 0.0
    state = env.reset()
    for t in range(max_time):
        if play:
            action = agent.get_best_action(state)
        else:
            action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
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
