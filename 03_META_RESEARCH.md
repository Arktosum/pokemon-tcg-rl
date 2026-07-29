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

## Phase 47: Greedy-Targeted PPO Results
- **Baseline (TITAN_TRANSFORMER_LEAGUE_01.pt):** 283/1000 wins (28.3% WR) vs GreedyAgent.
- **After 250 Greedy-PPO episodes (TITAN_GREEDY_PPO_01.pt):** 465/1000 wins (46.5% WR) — +18.2pp.
- **Architecture:** PokemonActorCritic (Transformer backbone, 120-dim state, 500-action space).
- **Hyperparameters:** lr=1e-4, clip=0.2, entropy_coef=0.05, value_coef=0.5, 4 PPO epochs/episode.
- **Diagnosis:** 250 episodes insufficient to reach 95% target. Entropy oscillation (0.09–0.90) suggests model is discovering but not stabilising optimal strategies.
- **Decision:** Pivot to Behavioral Cloning from top-Elo public replays — faster path to competitive Elo than continued PPO from random init.

## Phase 48: TOP_ELO_BC_MODEL Pipeline Design
- **Data Source:** Kaggle public replay API — every team's episodes are downloadable via `kaggle competitions episodes {sub_id}` + `kaggle competitions replay {ep_id}`.
- **Target Teams:** Top-10 leaderboard by Elo. Filter: MIN_SUB_SCORE = 1130.0 (hard cutoff).
- **Clean Dataset:** 6,480 episodes from 9 teams. wwwwwwww team excluded (both subs below 1130).
- **Skipped subs:** LiamK 1114.8, JZ 1061.5, Iliamna 1071.7, Yushin Ito 996.1, James Cox 654.4, titako0000 926.4, wwwwww 1126.2 & 1121.8.
- **BC Training Design:**
  - Parse each replay JSON → extract (obs_vector, action_taken) at every step where player == our agent's perspective.
  - Only include steps from the **winning player** to maximise signal quality.
  - Loss: CrossEntropy on action logits (policy head only for BC phase).
  - Model: TOP_ELO_BC_MODEL (same PokemonActorCritic architecture, fresh or warm-started from TITAN_GREEDY_PPO_01.pt).
- **Submission Strategy:** BC → submit → live Elo reading → PPO fine-tune only if Elo < 1000.
- **Key Risk:** Distribution shift — BC only saw states from top-vs-top games. May struggle vs weak bots on first few ladder matches before climbing to correct Elo bracket.

## Phase 70: League Reconnaissance (Kaggle Rule Differences)
Community discussions (`/competitions/pokemon-tcg-ai-battle/discussion/708586`) confirm several critical deviations in the Kaggle Simulator compared to official Pokémon TCG rules:
1. **Unselectable Attacks vs Declare-and-Fail**: In the simulator, if an attack effect can't be resolved (e.g. searching deck for Basic when bench is full), the attack is treated as *not selectable* from the beginning, instead of letting the player declare it and fail.
2. **Setup Phase Forced Benching Bug**: During setup, if you have multiple Basic Pokémon in hand, the simulator does not provide an "end turn" / "done" option, forcing the agent to bench *all* Basic Pokémon in hand.
3. **Mega Zygarde EX (Nullifying Zero)**: Target order is automatically resolved left-to-right; the player cannot choose.
4. **Mega Lopunny EX Bug**: `Gale Thrust` doesn't register the bonus 170 damage if promoted via a Pokémon ability (like Abra TWM) instead of a standard retreat.
5. **Telepathic Energy Bug**: Searches for *any* pokemon if attached to *any* type pokemon, and both searched pokemon go to hand.
6. **Prize-Taking Order**: Sequentially taken instead of simultaneously. (Irrelevant to outcome since "both players taking all prizes" = draw).
**Actionable Insight**: Our engine wrapper logic must prioritize the simulator's deterministic quirks (like returning `[]` when `maxCount == 0`) and be aware that agents might be forced to over-bench during setup. Rule-based agents must account for the Mega Lopunny damage bug if evaluating that card.
