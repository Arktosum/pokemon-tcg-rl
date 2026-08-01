# Phase 2: Neural Network Architecture & PPO Training Loop

The goal of this phase is to design the Neural Network architecture that consumes the 24-word state tensor produced by `parser.py` and outputs a policy distribution over the dynamic legal actions. We will also write a custom Reinforcement Learning (PPO) training loop from scratch to interface directly with the Kaggle `cabt` engine.

### Why Proximal Policy Optimization (PPO)?
Based on research into RL for Trading Card Games (TCGs), PPO combined with **Invalid Action Masking** is the standard approach for dynamic action spaces:
1. **Dynamic Action Spaces:** Unlike Q-learning (DQN), which requires evaluating $Q(s, a)$ for every possible action (a massive architectural bottleneck when the number of legal actions changes every turn), PPO is an Actor-Critic algorithm. This allows us to use an action pointer network and simply mask illegal actions by setting their logits to $-\infty$ right before the softmax layer. 
2. **Stochasticity:** TCGs involve high variance (shuffling, coin flips). PPO's clipped surrogate objective prevents destructive policy updates when a theoretically "good" move leads to a loss purely by chance, making training far more stable than DQN or A2C.

## Phase 2.1: Behavioral Cloning (Imitation Learning)
**Addressing the Replay Dataset:** We have 4,385 Kaggle replay files in our `dataset/matches` directory. PPO is notoriously sample-inefficient when starting from random weights in a complex environment. 
- **The Plan:** Before we run any RL, we will perform Behavioral Cloning (Supervised Learning). We will extract the states and actions from the *winning* agent in all 4,385 replays.
- We will train our neural network using Cross-Entropy loss to predict the winner's moves. This will "warm-start" the network, giving it a profound understanding of basic rules (attaching energy, playing supporters, attacking) so that our PPO loop starts with a competent agent rather than a random actor.

## Phase 2.2: Custom PPO Loop & Reward Shaping
**Addressing the Draw Conditions and Rewards:** 
In Pokémon TCG, a draw strictly occurs when both players meet a win condition at the exact same time (e.g., both take their last prize card simultaneously via recoil damage). 

*Correction on Step Limits*: I previously mentioned a "hard step limit" causing draws. A review of the Kaggle `cabt` engine source (`CABT_ENGINE_API_DOCUMENTATION.md`) proves this is practically impossible. The engine's JSON schema defines `'episodeSteps': 10000000` (10 million steps). However, it also enforces a `'runTimeout': 2000` (seconds) and a `'remainingOverageTime': 600` (seconds). If an agent stalls or loops excessively, it will hit the time limit long before the step limit. The API explicitly states: *"agent is disqualified with TIMEOUT status when this drops below 0."* A timeout is a loss, not a draw. 

Therefore, a draw is extremely rare and only happens through simultaneous win-states. While Kaggle assigns a 0 to draws (which simply moves our TrueSkill rating towards the mean), treating draws as strictly neutral in an RL loop often leads to passive, "turtling" behavior where the agent learns to stall rather than risk losing.

- **Reward Shaping Strategy:** We will write the PPO loop from scratch and implement a custom reward wrapper around the engine:
  - **Terminal Rewards:** Win = +1.0, Loss = -1.0, Draw = -0.1 (Slightly penalizing draws encourages decisive action over stalling).
  - **Dense Shaping (Early Training):** To help the agent understand progress, we will grant +0.05 for taking a Prize Card and -0.05 for losing a Prize Card. 
  - *Note: Once the agent is highly competent, we can anneal these dense rewards to zero and rely strictly on terminal sparse rewards for final fine-tuning.*

## Proposed Architecture

### 1. The Encoder (Transformer)
Based on `experiments/01_baseline/agent/parser.py`, the state is compressed into exactly 24 "words" of `D_model=128`. 
- We will inject **Learned Positional Embeddings** of size `[24, 128]` since the container index mathematically maps to a specific zone.
- Pass the sequence through 2-4 **Transformer Encoder Layers** with `nhead=4` or `8`.
- Apply Mean Pooling to generate a single **Global Board Context Vector** of size `[1, 128]`.

### 2. The Policy Head (Action Pointer Network)
- Take the Global Board Context Vector `[1, 128]` and perform **Dot-Product Attention** against the `[legal_action_count, 128]` action embeddings produced by the parser.
- This produces dynamic logits of size `[legal_action_count]`.
- Apply Invalid Action Masking (logits = $-\infty$) for impossible moves, then apply Softmax to generate the policy distribution $\pi(a|s)$.

### 3. The Value Head (Critic)
- Pass the Global Board Context Vector `[1, 128]` through an MLP to predict the state value $V(s) \in [-1, 1]$.

## Proposed Changes
- `experiments/02_ppo_training/model.py`: Implement the `TitanTransformer` in PyTorch.
- `experiments/02_ppo_training/behavioral_cloning.py`: Script to parse the `dataset/matches` and train the network via Supervised Learning.
- `experiments/02_ppo_training/env_wrapper.py`: Wrap the `cabt` engine to output our custom shaped rewards and handle vectorized self-play.
- `experiments/02_ppo_training/train_ppo.py`: Implement the Proximal Policy Optimization loop from scratch.

> [!IMPORTANT]
> **User Review Required: Ready for Execution?**
> The plan now includes Behavioral Cloning, custom reward shaping (penalizing draws), and a from-scratch PPO implementation.