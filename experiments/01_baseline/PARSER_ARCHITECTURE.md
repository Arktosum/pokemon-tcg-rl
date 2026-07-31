# State & Action Space Parser Architecture (TITAN v5.0 Baseline)

**Date**: 2026-07-30
**Component**: Step 1.2 State/Action Space Parsing
**Engine**: Kaggle `cabt` (Pokémon TCG)

## 1. Core Philosophy: The Hardcoding Fallacy vs. Semantic Stability
A purely dynamic parser (recursively crawling the JSON tree) outputs a varying number of features depending on the board state (e.g., 56 features on Turn 1, 60 features on Turn 5 when tools are attached). If the length and position of features shift dynamically, a neural network's weights lose semantic meaning.

Therefore, our parser is built on **Fixed Containers (Slots)** combined with a **Dynamic Bag (EmbeddingBag)**.

## 2. The 24-Word Sequence
To encode the board for a Transformer Encoder, we mathematically slice the physical table into exactly **24 distinct containers (words)**.

Each container receives a variable number of integer tokens (Card IDs, scalar fractions). PyTorch's `nn.EmbeddingBag(mode='sum')` looks up the $D_{model}$ embedding vector for each token and sums them together to produce exactly one fixed-size vector per container.

The Transformer reads a `[24, D_model]` tensor sequence ordered as follows:

| Word Index | Container / Zone | Explanation |
| :--- | :--- | :--- |
| `0-7` | Opponent's Bench (Slots 1-8) | Supports expanded formats padded to 8. Solves the **Attachment Problem** by keeping Pokémon and their attached tools/energies isolated. |
| `8-15` | Our Bench (Slots 1-8) | Same as above, for our bench. |
| `16` | Opponent's Active Spot | Contains the Active Pokémon ID, HP ratio, and all attached cards. |
| `17` | Our Active Spot | Contains our Active Pokémon and attachments. |
| `18` | Opponent's Player State | Contains deck count, discard pile IDs, prize counts, and status conditions. |
| `19` | Our Player State | Contains our deck count, discard pile IDs, prize counts, and status conditions. |
| `20` | Our Hand | Every Card ID in our hand is summed here. (Opponent's hand is hidden, so omitted). |
| `21` | Our Deck | Known / expected cards remaining. |
| `22` | Stadium | The currently active Stadium card. |
| `23` | Global State | Turn count, first player flag, and once-per-turn rules (`supporterPlayed`, `energyAttached`). |

## 3. Data Extraction Pipeline
We do not manually parse JSON strings. We use the engine's built-in C++ deserializer:
```python
from cg.api import to_observation_class
obs = to_observation_class(obs_dict)
```
This guarantees type-safe access to every deeply nested object (`Pokemon`, `PlayerState`, `State`) without `NoneType` crashes, exposing exhaustive features like `poke.tools` and `poke.energyCards`.

## 4. Why Order Does Not Matter
Because we use `EmbeddingBag(mode='sum')`, addition is commutative ($A + B = B + A$). Pushing `[Squirtle, WaterEnergy]` yields the exact same final vector as `[WaterEnergy, Squirtle]`. This perfectly encapsulates the unordered nature of attachments and hands.
