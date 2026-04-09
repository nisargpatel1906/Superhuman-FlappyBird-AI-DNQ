"""
DQN Agent for FlappyBird-v0
============================
Train:   python agent.py
Test:    python agent.py --test
"""

import os
import random
import itertools
import argparse
import numpy as np

import yaml
import torch
import torch.nn as nn
import torch.optim as optim

torch.backends.cudnn.benchmark = True  # Enable PyTorch hardware optimizations

import gymnasium as gym
import flappy_bird_gymnasium  # pip install flappy-bird-gymnasium

from experience_replay import ReplayMemory
from dqn_architecture import DQN


# ─────────────────────────────────────────────
# Helper: load parameters from YAML
# ─────────────────────────────────────────────
PARAMS_FILE = "parameters.yaml"
PARAM_SET   = "flappy_bird_v0"      # key inside the yaml


class DQNAgent:
    """
    Encapsulates the full DQN training / testing loop.
    """

    def __init__(self, is_training: bool = True, render: bool = False):
        self.is_training = is_training
        self.render      = render

        # ── Load hyper-parameters ──────────────────────────────────────────
        with open(PARAMS_FILE, "r") as f:
            all_params = yaml.safe_load(f)
        p = all_params[PARAM_SET]

        self.env_id             = p["env_id"]
        self.alpha              = p["alpha"]
        self.gamma              = p["gamma"]
        self.epsilon            = p["epsilon_init"] if self.is_training else 0.0
        self.epsilon_min        = p["epsilon_min"]
        self.epsilon_decay      = p["epsilon_decay"]
        self.replay_memory_size = p["replay_memory_size"]
        self.mini_batch_size    = p["mini_batch_size"]
        self.network_sync_rate  = p["network_sync_rate"]
        self.reward_threshold   = p["reward_threshold"]
        self.parameter_set      = p                         # keep full dict

        # ── Device Selection ───────────────────────────────────────────────
        self.device = torch.device("cuda") # FORCEFULLY USE GPU
        print(f"Using device: {self.device}")

        # ── Paths ──────────────────────────────────────────────────────────
        runs_dir = "runs"
        os.makedirs(runs_dir, exist_ok=True)
        self.log_file   = os.path.join(runs_dir, f"{p['log']}.log")
        self.model_file = os.path.join(runs_dir, f"{p['log']}.pt")

        # ── Loss / optimizer placeholders ─────────────────────────────────
        self.loss_fn   = nn.SmoothL1Loss() # Huber Loss instead of MSE
        self.optimizer = None   # initialised after networks are built

    # ──────────────────────────────────────────────────────────────────────
    def run(self):
        """Main entry-point: create env, networks and start episode loop."""

        # ── Vector Environment Setup ──────────────────────────────────────
        num_envs = 16 if self.is_training else 1
        
        if self.render and not self.is_training:
            # Single env for rendering test
            env = gym.make(self.env_id, render_mode="human")
        elif self.is_training:
            # Parallel vector env for fast training using multiple CPU threads
            env = gym.make_vec(self.env_id, num_envs=num_envs, vectorization_mode="async")
        else:
            env = gym.make_vec(self.env_id, num_envs=1, vectorization_mode="sync")

        state_dim  = env.single_observation_space.shape[0] if hasattr(env, 'single_observation_space') else env.observation_space.shape[0]
        action_dim = env.single_action_space.n if hasattr(env, 'single_action_space') else env.action_space.n

        # ── Policy network  (the one we train) ────────────────────────────
        policy_dqn = DQN(state_dim, action_dim).to(self.device)

        # ── Target network  (periodically synced copy) ────────────────────
        target_dqn = DQN(state_dim, action_dim).to(self.device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # ── Optimizer ─────────────────────────────────────────────────────
        self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

        # ── Replay memory ─────────────────────────────────────────────────
        memory = ReplayMemory(self.replay_memory_size)

        steps        = 0
        best_reward  = float("-inf")

        # ── Load existing model (for Testing AND Resuming Training) ───────
        if os.path.exists(self.model_file):
            print(f"Loading existing model from: {self.model_file}")
            # map location so we don't hit CPU/GPU mismatch issues
            policy_dqn.load_state_dict(torch.load(self.model_file, weights_only=True, map_location=self.device))
            target_dqn.load_state_dict(policy_dqn.state_dict())
            if not self.is_training:
                policy_dqn.eval()
        else:
            if not self.is_training:
                print(f"Error: No saved model found at '{self.model_file}'.")
                print("Please run training first (python agent.py) to generate the best model.")
                return

        # ── Episode loop (runs until manual Ctrl+C or reward threshold) ───
        
        # When using vector envs, reset returns batch of states
        states, _ = env.reset()
        states    = torch.tensor(states, dtype=torch.float32)

        # Track rewards for each parallel bird
        episode_rewards = np.zeros(num_envs)

        for global_step in itertools.count():
            
            # ε-greedy action selection for parallel envs
            if self.is_training and random.random() < self.epsilon:
                actions = env.action_space.sample() # Sample batch of actions
            else:
                with torch.no_grad():
                    # Handle single dimension state vs batched dimension states
                    if states.dim() == 1:
                        state_tensor = states.unsqueeze(0).to(self.device)
                    else:
                        state_tensor = states.to(self.device)
                    
                    actions = policy_dqn(state_tensor).argmax(dim=1).cpu().numpy()

            # For single env when testing with render
            if self.render and not self.is_training:
                action_to_take = actions[0] if isinstance(actions, np.ndarray) else actions
                next_state, reward, terminated, truncated, info = env.step(action_to_take)
                
                # Wrap single returns back into batch format for consistency
                next_states = np.array([next_state])
                rewards     = np.array([reward])
                terminations = np.array([terminated])
                truncations  = np.array([truncated])
                dones        = terminations | truncations
                actions      = np.array([action_to_take])
            else:
                # VectorEnv step
                next_states, rewards, terminations, truncations, infos = env.step(actions)
                dones = terminations | truncations

            # Update running rewards
            episode_rewards += rewards

            # Convert to tensors
            next_states_t = torch.tensor(next_states, dtype=torch.float32)
            rewards_t     = torch.tensor(rewards,     dtype=torch.float32)
            actions_t     = torch.tensor(actions,     dtype=torch.long)
            terminations_t = torch.tensor(terminations, dtype=torch.bool)

            if self.is_training:
                # Add each environment's step to memory
                for i in range(num_envs):
                    # Handle shapes correctly depending on if states is batched or single
                    s = states[i] if states.dim() > 1 else states
                    ns = next_states_t[i] if next_states_t.dim() > 1 else next_states_t
                    
                    # Do not store terminal states caused by truncation as true failures
                    is_true_terminal = bool(terminations_t[i])
                    memory.append((s.clone(), actions_t[i].clone(), rewards_t[i].clone(), ns.clone(), is_true_terminal))
                    
                steps += num_envs

                # Train when we have enough samples (optimized every 4 * num_envs steps for speed)
                if steps % (4 * num_envs) == 0 and len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self._optimise(mini_batch, policy_dqn, target_dqn)

                # Sync target network every N steps
                if steps % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            states = next_states_t

            # Handle completions in parallel array
            for i, done in enumerate(dones):
                if done:
                    final_reward = episode_rewards[i]
                    
                    if self.is_training:
                        # ε-decay happens per episode completion
                        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                        if final_reward > best_reward:
                            best_reward = final_reward
                            msg = (f"New best reward {best_reward:.1f} at step {steps}")
                            print(msg)
                            with open(self.log_file, "a") as f:
                                f.write(msg + "\n")
                            torch.save(policy_dqn.state_dict(), self.model_file)
                    
                    print(f"Step {steps:>8}  |  "
                          # Env N indicator removed for single instances
                          f"Env_{i} Reward: {final_reward:>8.1f}  |  "
                          f"Epsilon: {self.epsilon:.4f}")
                          
                    episode_rewards[i] = 0.0 # reset for this bird's next run

    # ──────────────────────────────────────────────────────────────────────
    def _optimise(self, mini_batch, policy_dqn, target_dqn):
        """
        One gradient-descent step on a batch of experiences.

        For each experience (s, a, r, s', done):
          - If done : target_Q = r
          - Else    : target_Q = r + γ · max_a'[ target_dqn(s') ]
        Then minimise SmoothL1Loss(target_Q, policy_dqn(s)[a]).
        """
        states, actions, rewards, next_states, terminations = zip(*mini_batch)

        states      = torch.stack(states).to(self.device)
        actions     = torch.stack(actions).to(self.device)
        rewards     = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        terminations = torch.tensor(terminations, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # max Q-value for next states from the TARGET network
            next_q_max = target_dqn(next_states).max(dim=1).values

        # Bellman target
        target_q = rewards + self.gamma * next_q_max * (1.0 - terminations)

        # Q-values predicted by the POLICY network for taken actions
        current_q = policy_dqn(states).gather(
            dim=1, index=actions.unsqueeze(1)
        ).squeeze(1)

        # Compute loss & back-propagate
        loss = self.loss_fn(current_q, target_q.unsqueeze(1) if current_q.dim() != target_q.dim() else target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN FlappyBird Agent")
    parser.add_argument("--test", action="store_true",
                        help="Run in testing mode (omit for training default)")
    args = parser.parse_args()

    is_training = not args.test
    render      = not is_training   # Render visually only when testing

    agent = DQNAgent(is_training=is_training, render=render)
    agent.run()
