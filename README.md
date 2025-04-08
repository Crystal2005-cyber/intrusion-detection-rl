# 🚨 Intrusion Detection System using Reinforcement Learning (Q-Learning)

This project demonstrates a simple **Reinforcement Learning-based Intrusion Detection System (IDS)** using the **NSL-KDD dataset** and **Q-learning**. The model learns to classify network traffic as various types of attacks or normal behavior.

---

## 📁 Project Structure

intrusion-detection-rl/ ├── data/ # Raw dataset files (download manually) │ ├── KDDTrain+.TXT │ └── KDDTest+.TXT ├── preprocess.py # Preprocesses and saves X_train, y_train, etc. ├── rl_agent.py # Q-learning agent implementation ├── environment.py # RL environment simulating classification ├── train.py # Script to train the RL agent ├── requirements.txt # Required Python libraries └── README.md 