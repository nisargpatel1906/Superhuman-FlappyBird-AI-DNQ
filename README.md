# Superhuman FlappyBird AI (DQN) 🐦

[![Stars](https://img.shields.io/github/stars/nisargpatel1906/Superhuman-FlappyBird-AI-DQN?style=flat-square&color=blue)](https://github.com/nisargpatel1906/Superhuman-FlappyBird-AI-DQN/stargazers)
[![Forks](https://img.shields.io/github/forks/nisargpatel1906/Superhuman-FlappyBird-AI-DQN?style=flat-square&color=blue)](https://github.com/nisargpatel1906/Superhuman-FlappyBird-AI-DQN/network/members)
[![Issues](https://img.shields.io/github/issues/nisargpatel1906/Superhuman-FlappyBird-AI-DQN?style=flat-square&color=blue)](https://github.com/nisargpatel1906/Superhuman-FlappyBird-AI-DQN/issues)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10_|_3.12-blue.svg?style=flat-square&logo=python&logoColor=white)

An incredibly optimized **Deep Q-Network (DQN)** agent that learns to play **FlappyBird** at a superhuman level using Reinforcement Learning.

---

## ⚡ High-Performance Architecture

To break past standard Python CPU/GPU execution bottlenecks, this codebase was heavily optimized:

1. **In-Loop Dynamic Optimization:** Instead of waiting for an entire episode (game over) to train a single batch, the agent calls `_optimise()` constantly *inside* the live game loop. The neural network's weights are dynamically updated frame-by-frame, resulting in immense sample efficiency.
2. **Asynchronous Vectorized Environments:** The Flappy Bird simulation runs exclusively on the CPU, heavily bottlenecking the GPU. To fix this, we utilize `gym.make_vec(..., vectorization_mode="async")` to spawn 16 parallel Flappy Bird games mapped directly to the hardware threads of modern CPUs. A single batch of 16 states is generated simultaneously, preventing the GPU from ever idling.

## 📈 Training Results
- **Hardware:** Intel Core i5-13450HX (16-thread), RTX 3050 (CUDA)
- **Total Steps Trained:** ~5.26 Million Steps
- **High Score Milestones:**
  - **101 Score:** Achieved at **2.9 Million steps**.
  - **1000 Score:** Resumed training pushed the model to **5.26 Million steps**, ultimately reaching a flawless high score of **1000**!

---

## 📂 Project Structure

```text
├── agent.py               # Main DQN agent (training + testing logic)
├── dqn_architecture.py    # Neural network policy & target networks
├── experience_replay.py   # Replay memory (FIFO deque)
├── parameters.yaml        # Centralized hyper-parameter configurations
├── requirements.txt       # Project dependencies
└── runs/                  # Auto-created: saved models (.pt) + logs
```

## 🛠️ Quick Setup

> **⚠️ IMPORTANT:** This project strictly requires **Python 3.10** or **Python 3.12**. Other versions may cause compatibility issues with the Gymnasium environments or PyTorch.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nisargpatel1906/Superhuman-FlappyBird-AI-DQN.git
   cd Superhuman-FlappyBird-AI-DQN
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv_rl
   
   # Windows
   .\venv_rl\Scripts\activate
   # Linux/Mac
   source venv_rl/bin/activate
   ```

3. **Install Dependencies:**
   Install the required packages from the generated `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   > *Note: For optimal performance, ensure you install [PyTorch with CUDA support](https://pytorch.org/get-started/locally/) if you have an NVIDIA GPU.*

## 🚀 Usage

### 🏋️‍♂️ Training the Agent
Run the main script to start training. 
- It spawns **16 staggered async environments**.
- The best model is **auto-saved** to `runs/flappy_bird_v0.pt` whenever a new high score is reached.
- **Auto-Resume:** If you restart the script, it automatically loads the `.pt` file so you never lose progress.

```bash
python agent.py
```

### 🎮 Testing the Agent (Watch it play!)
Loads the saved best model (`runs/flappy_bird_v0.pt`) and renders the game visually with a single environment while locking exploration (`Epsilon`) to `0.0`.

```bash
python agent.py --test
```

---

## ⚙️ Key Concepts & Hyperparameters

Configured directly in `parameters.yaml`.

| Concept/Parameter | Value | Details |
|---|---|---|
| **Epsilon Init** | 1.0 | Starts at 100% random exploration |
| **Epsilon Min** | 0.05 | Never goes below 5% exploration during training |
| **Epsilon Decay** | 0.995 | Exponentially decays per completed episode |
| **Replay Memory** | 100,000 | Max experiences stored in memory deque |
| **Mini Batch Size** | 256 | Samples evaluated per GPU step |
| **Network Sync Rate** | 10 | Frame steps between target network synchronizations |
| **Optimizer** | Adam | Optimizes model weights (`alpha: 0.001` learning rate) |
| **Loss Function** | SmoothL1Loss | Huber loss replaced MSE for resilient gradient updates |
| **Device Execution** | CUDA | `torch.backends.cudnn.benchmark = True` enabled for speed |

---

*Developed with ❤️ by [Nisarg Patel](https://github.com/nisargpatel1906)*
