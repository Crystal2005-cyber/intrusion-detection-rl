import numpy as np

class QLearningAgent:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.n_actions = n_actions

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)  # explore
        else:
            return np.argmax(self.q_table[state])     # exploit

    def update(self, state, action, reward, next_state):
        best_next_action = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_action
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta
