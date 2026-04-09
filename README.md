# DQN FlappyBird Agent

A Deep Q-Network (DQN) agent that learns to play **FlappyBird** using reinforcement learning.

## Project Structure

```
├── agent.py               # Main DQN agent (training + testing)
├── dqn_architecture.py    # Neural network (policy & target networks)
├── experience_replay.py   # Replay memory (FIFO deque)
├── parameters.yaml        # All hyper-parameters
└── runs/                  # Auto-created: saved models + logs
```

## Setup

We created a custom virtual environment (`venv_rl`) with PyTorch CUDA to maximize GPU usage:
```bash
python -m venv venv_rl
.\venv_rl\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.\venv_rl\Scripts\python.exe -m pip install gymnasium flappy-bird-gymnasium pyyaml numpy
```

## Training

```bash
.\venv_rl\Scripts\python.exe agent.py
```

- Runs **16 staggered environments asynchronously** allowing the CPU to efficiently feed the GPU.
- Best model is auto-saved to `runs/flappy_bird_v0.pt` whenever a new high score is reached.
- **Auto-Resume:** If you restart the script, it automatically loads the `.pt` file so you never lose progress!

## Testing (watch the agent play)

```bash
.\venv_rl\Scripts\python.exe agent.py --test
```

Loads the saved best model and renders the game visually with a single environment and `Epsilon` locked to `0.0`.

---

## High-Performance Architecture Optimization

To break past the standard Python CPU/GPU execution bottlenecks, the codebase was heavily rewritten to include two major optimizations:

1. **Option 1: In-Loop Dynamic Optimization:** Instead of waiting for an entire episode (game over) to train a single batch, the agent now calls `_optimise()` every 4 frame steps *inside* the live game loop. The neural network's weights are dynamically updated, and it learns frame-by-frame resulting in immense sample efficiency.
2. **Option 3: Asynchronous Vectorized Environments:** The Flappy Bird simulation runs exclusively on the CPU, heavily bottlenecking the RTX 3050 GPU. To fix this, we utilized `gym.make_vec(..., vectorization_mode="async")` to spawn 16 parallel Flappy Bird games mapped directly to the 16 hardware threads of the Intel Core i5-13450HX CPU. A single batch of 16 states is generated simultaneously preventing the GPU from ever idling.

## Training Results
- **Hardware:** Intel Core i5-13450HX (16-thread), RTX 3050 (CUDA)
- **Total Steps Trained:** ~5.26 Million Steps
- **High Score Milestones:**
  - **101 Score:** Achieved at **2.9 Million steps** (training was manually stopped here and evaluated).
  - **1000 Score:** We then resumed training, pushing the model to **5.26 Million steps**, ultimately reaching a flawless high score of **1000**!

---

## Key Concepts & Hyperparameters

| Concept/Parameter | Value | Details |
|---|---|---|
| **Epsilon Init** (`epsilon_init`) | 1.0 | Starts at 100% random exploration |
| **Epsilon Min** (`epsilon_min`) | 0.05 | Never goes below 5% exploration during training |
| **Epsilon Decay** (`epsilon_decay`) | 0.995 | Exponentially decays per completed episode |
| **Replay Memory** | 100,000 | Max experiences stored in memory deque |
| **Mini Batch Size** (`mini_batch_size`) | 256 | Samples evaluated per GPU step |
| **Network Sync Rate** (`network_sync_rate`) | 10 | Frame steps between target network synchronizations |
| **Optimizer** | Adam | Optimizes model weights (`alpha: 0.001` learning rate) |
| **Loss Function** | SmoothL1Loss | Huber loss replaced MSE for resilient gradient updates |
| **Device Execution** | CUDA | `torch.backends.cudnn.benchmark = True` enabled for speed |
