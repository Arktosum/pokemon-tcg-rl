# Pokémon TCG RL: Master Findings Archive
*Compiled prior to Repository Wipe.*

## 1. The Kaggle Meta & Strategy
- **The ELO Bar:** Public rule-based bots are incredibly strong. The `Alakazam 5th Place` and `Archaludon/Cinderace` bots reliably achieve 850+ ELO.
- **The Core Directive:** Any RL model built going forward **MUST** be aggressively tested against these top-tier bots to prove its worth. Fighting generic heuristic bots or random policies is insufficient. Fictitious self-play without anchoring against the Kaggle meta leads to mode collapse.
- **Deck Specialization:** Generalized RL models fail due to the massive combinatorial complexity of the TCG. The optimal approach is to freeze the environment to a single deck archetype (e.g., Psychic/Alakazam) to restrict the action space and force the model to learn deep tactical strategies over generalized shallow rules.

## 2. Infrastructure & Stability (The Watchdog)
- **Engine Volatility:** The underlying C++ Engine (`cg.dll`) used by Kaggle is highly unstable when subjected to massive parallelization. It is prone to infinite loops and Segmentation Faults that will silently crash the Python interpreter.
- **The Solution:** A **Native Asynchronous Watchdog**. All environments must be isolated via `multiprocessing.Process` with a strict `timeout` (e.g., 1.5s). If a worker process hangs or throws an `EOFError`, the pipe must be explicitly closed, the `pid` killed, and the worker resurrected with a dummy observation returned to the trainer to skip the frame.

## 3. RL Architecture (Transformers & PPO)
- **State Representation:** The game state is wildly variable. A purely dense network fails. The optimal architecture uses a Transformer Encoder to process a dynamic sequence of cards across arbitrary zones (Hand, Deck, Bench, Active, Discard).
- **Action Space:** Actions are represented as `(context, option, indices)`. Because the Kaggle engine asks the agent to "select" from dynamic arrays of options (1 to N), standard discrete action spaces do not work. 
- **Set-Based Decoding:** The model must process the *valid options array* through a cross-attention Transformer Decoder to evaluate the relative expected value of each valid target.
- **PPO Tweaks:** Global value clipping, per-group advantage centering, and massive batch sizes (16,384+ steps) are required to combat the extreme variance in TCG reward signals.

## 4. Behavioral Cloning (The Bootstrapper)
- Pure RL starting from random weights is mathematically unfeasible within a standard timeframe due to the sparse rewards of winning a 100-turn game.
- Extracting raw battle replays from Kaggle Notebooks and converting them into state-action tensors allows us to train a Behavioral Cloning (BC) model.
- BC initializes the policy network with a profound understanding of game logic, acting as the mandatory foundation for PPO scaling.

---
*End of Archive. The repository is slated for deletion.*
