# 03_META_RESEARCH
## 1. DOMAIN UNDERSTANDING
Pokemon TCG: Active pokemon, bench, hand, energy attachments, prize cards.
## 2. EVALUATION METRIC
Kaggle leaderboard rating (Goal: 1100+). 
*Note (Phase 46 Correction):* The initial submission score (e.g., 600.0) is merely the result of a Validation Episode (crash-check), NOT a measure of model skill. True rankings are generated over time in the matchmaking pool.
Vectorized state-space representation encoding game board arrays for Neural Network.

### Evidence-Backed V2 State Vectorization Blueprint (Empirical Grounding)
* **Design Philosophy**: Based on RLCard and MageZero, we must preserve semantic geometry rather than naively flattening. We will output an array of Token IDs (for `nn.Embedding`) and scalars (normalized).
* **Global Game State**: Turn number, turnActionCount, player index, firstPlayer.
* **Player State (x2)**:
  * Handcrafted Features: deckCount, handCount, benchMax, prizeCount, status conditions (poisoned, burned, asleep, paralyzed, confused).
  * Active Pokemon: Card ID (integer for embedding), HP, MaxHP, Attached Energy Count.
  * Bench Pokemon (Fixed size 5): Array of [Card ID, HP, MaxHP, Energy Count]. Zero-padded if empty.
* **Total Vector Size**: Flattened array of shape (120,) containing integers and floats. High boundary set to 3000 to accommodate max Card ID (2104).
* **Action Space**: `Discrete(500)`. Strict Dynamic Masking applied (`action_mask`) as per RLCard standards to prune illegal moves. If `maxCount > 1`, randomly sample remaining from unmasked pool to fulfill engine requirements.

### Phase 3 Architecture Blueprint
* **PokemonAlphaNet**: A Two-Stream Feature Extractor.
  * **Stream A**: `nn.Embedding(3000, 32)` extracting the 12 categorical Card IDs from the state.
  * **Stream B**: `Linear -> LayerNorm -> SiLU` processing the remaining 58 continuous/scalar game state features.
  * **Backbone**: Streams A and B are concatenated (`torch.cat`) and fed into a Residual MLP backbone.
* **Dual Heads**: 
  * **Policy Head**: Emits 500 logits. Illegal logits masked to $-1e9$ before applying Softmax.
  * **Value Head**: Emits a scalar $[-1, 1]$ via `Tanh`.
* **PUCT Search**: ISMCTS-compatible tree node structure utilizing Predictor + Upper Confidence Bound applied to Trees, injecting Dirichlet noise ($\alpha=0.3$) at the root node for exploration.

### Phase 4 Training Pipeline & Loss Engineering Blueprint
* **Self-Play Trajectory Target**: Store tuples of $(s_t, \boldsymbol{\pi}_t, z_t)$, where $s_t$ is the environment state vector, $\boldsymbol{\pi}_t$ is the MCTS search-improved policy vector, and $z_t \in \{-1, +1\}$ is the final game winner relative to the acting player at time $t$.
* **AlphaZero Combined Loss Function**: 
$$L = (z - v)^2 - \boldsymbol{\pi}^T \log \mathbf{p} + c \Vert{}\theta\Vert{}^2$$
where $v$ is the predicted value, $\mathbf{p}$ is the predicted policy vector, and $c \Vert{}\theta\Vert{}^2$ is L2 weight decay ($c = 10^{-4}$).

### Phase 5 Behavioral Cloning Blueprint
* **Data Source**: High-scoring Kaggle simulation `.json` episode replays.
* **State Translation**: Extract JSON dictionaries and translate them into the exact V2 (120,) state vector format and (500,) action mask to prevent State-Space Mismatch.
* **Supervised Target**: The actual action taken by the winning player ($\mathbf{p}_{expert}$) and the final game result ($z_{expert}$).
* **BC Loss**: Minimize Cross-Entropy (or KL Divergence) for the Policy Head and MSE for the Value Head: $L_{BC} = MSE(\hat{z}, z_{expert}) - \sum \mathbf{p}_{expert} \log \hat{\mathbf{p}}$.


## Phase 13: Proximal Policy Optimization (PPO)

Due to Kaggle C++ engine state cloning limitations, MCTS is structurally impossible. We pivot to PPO. The network is renamed PokemonActorCritic. Rollouts are stored in PPOBuffer, Generalized Advantage Estimation (GAE) is applied, and weights are updated via PPO Clipped Objective.


## Reward Shaping (Phase 16)
- **Win:** +1.0
- **Loss:** -1.0
- **Dense Rewards (Irreversible Progression):** Taking a Prize Card = +0.1, Opponent taking Prize Card = -0.1.

## Residual Network Backbone (Phase 17)
- **Architecture:** Replaced MLP with 4 Residual Blocks (Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> Skip Add -> ReLU).
- **Hidden Dimension:** 256.
- **Memory Management:** Aggressive gc.collect() and empty_cache() during PPO rollouts to prevent VRAM OOM.

### Phase 18 Curriculum Rewards
- **Dense Progression Rewards:** Add +0.05 for Bench-filling (max 5) and +0.05 for Energy Attachment (max 3 per mon).
- **The Safety Lock:** Implemented via max total bounds per episode to prevent infinite loop reward farming.
- **Robustness Training (Domain Randomization):** 30% of self-play games are against a Random Agent to force OOD robustness.
