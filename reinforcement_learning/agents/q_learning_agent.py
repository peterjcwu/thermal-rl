import os
import math
import random
import pickle
from collections import defaultdict


class QLearningAgent:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.q_table = defaultdict(lambda: defaultdict(lambda: 0))
        self.gamma = 0.99
        self.episode_count = 0
        self.model_path = "model.pickle"

    def get_value(self, state):
        """
        Compute agent's estimate of V(s) using current q-values
        V(s) = max_over_action Q(state,action)
        """
        max_q_val = float("-inf")
        for action in range(self.n_actions):
            q_val = self.q_table[state][action]
            if q_val > max_q_val:
                max_q_val = q_val
        return max_q_val

    def update(self, s, a, r, s_) -> None:
        """
        Q(s,a) := (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))
        """
        q_value = self.q_table[s][a]
        self.q_table[s][a] = (1 - self.alpha) * q_value + self.alpha * (r + self.gamma * self.get_value(s_))

    def get_best_action(self, state):
        best_action = None
        max_q_val = float("-inf")
        for action in range(self.n_actions):
            q_val = self.q_table[state][action]
            if q_val > max_q_val:
                max_q_val = q_val
                best_action = action
        return best_action

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.n_actions))
        else:
            return self.get_best_action(state)

    @property
    def alpha(self):
        return 0.5
        # return max(0.01, min(0.5, 1.0 - math.log10((self.episode_count + 1) / 25)))

    @property
    def epsilon(self):
        return 0.25 * (0.99 ** (self.episode_count * 2))
        # return max(0.01, min(1.0, 1.0 - math.log10((self.episode_count + 1) / 25)))

    def load_weight(self, weight):
        if not os.path.isfile(self.model_path):
            return
        with open(self.model_path, "r") as f_in:
            self.q_table = pickle.load(f_in)

    def dump_weight(self):
        with open(self.model_path, "w") as f_out:
            pickle.dump(f_out, self.q_table)
