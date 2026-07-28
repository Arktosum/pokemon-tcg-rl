# 05_RAW_RESEARCH_ARCHIVE

**SYSTEM RULE:** APPEND ONLY. 

## RAW SNIPPETS & DATA DUMPS

### Web Search Results: PTCG / MTG / Hearthstone State Vectorization (2026-07-28)

*   **Handling Raw vs. Structured Data:** Convert non-numeric data to numeric form: Encode card IDs, types, and statuses into categorical embeddings or one-hot vectors.

*   **Flattening vs. Structural Modeling:** Advanced approaches use Transformer-based architectures or specialized subnets that treat specific game elements (e.g., active Pokémon, benched Pokémon, hand, energy) as distinct tokens or sequences.

*   **Legal Action Masking:** A crucial part of the state representation. RL agents use action masking—where the output layer of the policy network is masked to assign zero probability to illegal actions.

*   **Dynamic Hashing & One-Hot Encoding:** To handle the sparse and open-ended nature of card games, developers often use dynamic feature hashing or one-hot encoding for specific card IDs (MageZero, RLCard).

*   **Hidden Information:** Effective agents maintain an "implicit belief state".

*   **Domain-Specific Features:** Indicators for "super-effective" moves, summary statistics of both players' board states, and relative counts of resources (Energy, HP levels).

### Web Search Results: AlphaZero Dual-Head & ISMCTS (2026-07-28)

*   **Mixed Data Trunks:** PyTorch implementations split inputs: `nn.Embedding` for categorical tokens (Card IDs) and `nn.Linear` for continuous scalars. They concatenate the outputs (`torch.cat`) into a shared ResNet or MLP trunk.

*   **Dual Heads:** Post-trunk, the network forks. Policy Head: `Linear -> Softmax` (with dynamic masking setting illegal logits to `-1e9`). Value Head: `Linear -> Tanh` to bound expected outcomes between [-1, 1].

*   **ISMCTS & PUCT:** In imperfect information card games, ISMCTS explores an "information set tree" via determinization. PUCT (`pb_c * prior * sqrt(N) / (1 + n)`) balances NN prior exploitation against exploration. The Kaggle `ptcg_engine` explicitly provides `search_begin()` and `search_step()` to spawn branchable search states for this exact purpose.

### Web Search Results: PTCG Kaggle Replay Parsing (2026-07-28)

*   **Data Source:** Top-rated Kaggle JSON episodes can be downloaded manually via the daily datasets ("PTCG AI Battle Challenge Simulation Episodes") or programmatically using the Kaggle API.

*   **JSON Structure:** 

    *   `configuration`: Simulation params (seed, steps).

    *   `info`: Metadata, `EpisodeId`, agent names.

    *   `rewards`: Final match outcomes (1, 0, -1).

    *   `steps`: Step-by-step game log.

*   **State Translation:** The main challenge is converting the raw JSON `steps` dictionary arrays into the normalized V2 `(120,)` state representation matching the live `env.py`. Strict mapping prevents State-Space Mismatch.

### Web Search Results: PTCG Heuristic Greedy Agents (2026-07-28)

*   **Performance Benchmark:** Used to establish a floor. A simple rule-based agent employs greedy logic - prioritizing moves that deal the most immediate damage, attaching energy to active Pokemon, or evolving whenever possible.

*   **Implementation:** They evaluate based on immediate metrics. For our environment, we prioritize non-Pass actions (e.g., Action > 0), strictly filtering through `action_mask` to prevent infinite loops (trying to play cards we can't afford).

