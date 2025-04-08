import pandas as pd
from rl_agent import QLearningAgent
from environment import IntrusionEnv

# Load preprocessed data
X_train = pd.read_csv("X_train.csv").values
y_train = pd.read_csv("y_train.csv")['label'].values

env = IntrusionEnv(X_train, y_train)

# Define agent
n_states = X_train.shape[1]
n_actions = len(set(y_train))
agent = QLearningAgent(n_actions=n_actions, n_states=100)

# Training loop
episodes = 10
for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        if next_state is not None:
            agent.update(state, action, reward, next_state)
            state = next_state
        total_reward += reward

    print(f"Episode {ep+1}: Total reward = {total_reward}")
