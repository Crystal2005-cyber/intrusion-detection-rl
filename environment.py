import numpy as np

class IntrusionEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.current_idx = 0
        self.n_samples = len(X)

    def reset(self):
        self.current_idx = 0
        return self._get_state()

    def _get_state(self):
        return int(np.argmax(self.X[self.current_idx]))

    def step(self, action):
        true_label = self.y[self.current_idx]
        reward = 1 if action == true_label else -1
        self.current_idx += 1
        done = self.current_idx >= self.n_samples
        next_state = self._get_state() if not done else None
        return next_state, reward, done
