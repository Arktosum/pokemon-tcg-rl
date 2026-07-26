# 03_META_RESEARCH

## SUMMARY OF GROUND TRUTH
The Kaggle Simulation competition requires building an AI for the `cabt Engine`. Importantly, Kaggle provides daily exports of top-rated episode replays to facilitate Behavior Cloning (BC) and Reinforcement Learning (RL). The dataset contains 60 files (327 MB) including the engine source code.

## 1. COMPETITION ANALYSIS (Based on `00_GROUND_TRUTH.md`)
Before discussing algorithms, we must rigorously analyze the competition parameters as defined by Kaggle and The Pokémon Company.

### 1.1 Core Game Mechanics & Challenges
- **Imperfect Information (Hidden State):** The overview explicitly states players must be "mindful of their opponent's own strategies, decks and hands," and highlights that "Not knowing what cards an opponent holds presents a core challenge." This officially confirms the environment is a POMDP (Partially Observable Markov Decision Process).
- **Stochasticity (Randomness):** The rules emphasize that "card draws and coin tosses introduce additional gameplay variables." A successful agent cannot rely on deterministic planning (like perfect Monte Carlo Tree Search).
- **Complexity:** The agent must consider "various Pokémon types and thousands of different card combinations" and "evolving strategies."

### 1.2 The Simulator Engine (`cabt Engine`)
- **API Inputs:** Every turn, the agent receives an `Observation` containing:
  - Game logs (history of the match)
  - Current board state (Public zones, own hand, own deck count)
  - A strict list of **Legal Options**
- **API Outputs:** The agent must return an array of integers representing the *indices* of the options it chooses to execute.
- **Rule Enforcement:** The engine strictly handles rule enforcement—it "only ever presents legal moves," meaning our agent does not need to learn the rules of the game from scratch, it only needs to learn *which* legal move is best.

### 1.3 Evaluation & Leaderboard System
- **Ranking Algorithm:** Kaggle uses a Gaussian Skill Rating system $N(\mu, \sigma^2)$ (very similar to Microsoft's TrueSkill or Elo with confidence intervals). All agents start at $\mu_0 = 600$.
- **Matchmaking:** The system runs continuous episodes, matching agents with similar ratings.
- **Submission Limits:** You can submit 5 agents per day. Submissions must be a `.tar.gz` containing a `main.py` and a `deck.csv`. Only the 2 most recent submissions are kept active on the ladder.
- **Validation:** Before entering the ladder, a submission must pass a Validation Episode by playing against a copy of itself. If it crashes, it fails.

### 1.4 Provided Data
- **Engine Source:** The `ptcg_engine` source code and `sample_submission`.
- **Card Metadata:** `EN_Card_Data.csv` (35 columns including HP, Type, Weakness, Resistance, Move Costs, Damage, Rules). This is crucial for giving our neural network mathematical representations of the cards (embeddings).
- **Replay Data:** Kaggle provides daily exports of top-rated episodes in the forums to facilitate "BC/RL/IL" (Behavior Cloning / Reinforcement Learning / Imitation Learning).

### 1.5 ONLINE RESEARCH (COMMUNITY META)
Based on web queries to Kaggle Discussions and `kaggle kernels list -c pokemon-tcg-ai-battle`, here is what the community is actually doing:
1. **Rule-Based Heuristics Dominate Early Meta:** The most upvoted notebooks on Kaggle are almost entirely hardcoded, rule-based agents tailored to specific deck archetypes. Examples include: *"A Sample Rule-Based Agent Mega Lucario ex Deck"*, *"Rule-based, not psychic: Alakazam (Best: 5th)"*, and *"Beating the Day-1 #1 Crustle Bot"*. This proves that standard "out-of-the-box" Reinforcement Learning is currently losing to well-crafted, deck-specific logic trees.
2. **RL/MCTS is being Explored:** The #1 most upvoted notebook overall is *"Reinforcement Learning and MCTS sample code"* by Kiyota, proving the community is actively trying to crack the RL problem, but the abundance of heuristic bots on the leaderboard indicates RL hasn't fully solved the vast action space yet.
3. **Score Stabilization Strategy:** Forum discussions reveal a quirk in Kaggle's TrueSkill system: newer agents get scheduling priority, but submitting more than 2 agents at a time causes them to steal matches from each other, drastically slowing down score stabilization. We must submit sparingly.
4. **Archetype Analysis:** The community actively analyzes the daily replay files (e.g., *"Replay Archetype Analysis"* notebooks) to figure out which decks (Alakazam, Archaludon, Starmie) have the highest win rates, allowing them to hardcode counters.

### 1.6 GROUND TRUTH CODE ANALYSIS (Community Algorithms)
Based on our extraction and reading of the top Kaggle notebooks saved in `input/ground_truth/notebooks/`:
- **Kiyota's RL + MCTS:** We discovered that parsing the JSON `Observation` space is incredibly difficult due to variable list lengths (bench size, hand size). Kiyota solved this elegantly using a PyTorch `EmbeddingBag` combined with a **Transformer Encoder-Decoder** architecture. The Encoder maps the board into a latent space, and the Decoder parses legal moves. We will adopt this exact tensor encoding scheme to avoid dimension mismatch errors when building our RL architecture.
- **Roman Rozen's Baseline:** The top rule-based bot uses a massive heuristic scoring tree (e.g., `target_score`, `prize_count`) combined with a lightweight UCB1 (Upper Confidence Bound) search to simulate one step ahead and avoid traps. This proves that raw heuristics augmented with mini-search trees are the current standard to beat.

## 2. ALGORITHM JUSTIFICATION (Why PPO?)
Before proceeding, we must scientifically justify the choice of Proximal Policy Optimization (PPO) over other Reinforcement Learning algorithms.

**1. The POMDP Problem (Hidden Information)**
Pokémon TCG is a Partially Observable Markov Decision Process (POMDP). You do not know the opponent's hand, the exact order of your deck, or your prize cards.
* **Why not DQN?** Deep Q-Networks (DQN) evaluate the exact value of a specific state-action pair ($Q(s,a)$). In POMDPs, the true state $s$ is unknown, which breaks the Markov property that DQN relies on, leading to severe instability.
* **Why PPO?** Policy Gradient methods like PPO directly optimize the policy $\pi(a|s)$ rather than the value function. When combined with an LSTM/GRU layer to maintain memory of past observations (RNN-PPO), it naturally handles hidden information.

**2. Variable and Massive Action Spaces**
At any given turn, a player might have 0 actions or 50 valid actions depending on their hand and bench.
* **Why not DQN?** DQN requires outputting a Q-value for *every possible action* in the game's entire action space, masking out invalid ones. This is extremely inefficient.
* **Why PPO?** PPO's actor network outputs a probability distribution over valid actions. It scales elegantly to massive action spaces through action masking.

**3. Self-Play Stability (The Clipping Objective)**
We will be training the agent via Self-Play (playing against past versions of itself).
* **Why not AlphaZero/MCTS?** AlphaZero relies on a perfect forward model (Monte Carlo Tree Search). TCGs have high stochasticity (coin flips, card draws) making MCTS computationally prohibitive due to extreme branching factors.
* **Why PPO?** PPO uses a clipped surrogate objective function: $L^{CLIP}(\theta) = \hat{E}_t [ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) ]$. This mathematically guarantees that the policy does not change too drastically in a single update. In self-play, this prevents "catastrophic forgetting" where the agent unlearns good behavior by overfitting to a specific weakness in its current opponent.
