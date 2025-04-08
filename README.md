# 🚨 Intrusion Detection System using Reinforcement Learning (Q-Learning)

This project demonstrates a simple Reinforcement Learning-based Intrusion Detection System (IDS) using the NSL-KDD dataset and Q-learning. The model learns to classify network traffic as attacks or normal behavior.

---

## 📁 Project Structure

intrusion-detection-rl/ ├── data/ 
# Raw dataset files (download manually)
                                        │ ├── KDDTrain+.TXT
                                        │ └── KDDTest+.TXT
  # Preprocesses and saves X_train, y_train, etc. 
                                        ├── preprocess.py  
  # Q-learning agent implementation 
                                        ├── train.py
  # RL environment simulating classification
                                        ├── environment.py
  # Script to train the RL agent 
                                        ├── rl_agent.py 
  # Required Python libraries 
                                        ├── requirements.txt
  # Description                                      
                                        └── README.md 
