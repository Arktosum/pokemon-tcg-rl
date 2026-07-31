# 03_META_RESEARCH
## 1. DOMAIN UNDERSTANDING
- **Game Engine**: `cabt` (runs via `kaggle-environments`).
- **Input**: Agent receives an `Observation` containing game logs, current board state, and a list of legal options.
- **Output**: Agent returns a list of indices corresponding to the selected options.
- **Differences from Official Rules**: Some attacks are not selectable if the effect cannot be fully resolved (e.g., drawing cards when the deck is empty). Target order for some attacks (like Mega Zygarde ex) is automatic. Prize taking order on simultaneous Knock Outs differs slightly but resolves to a draw if both take all prizes.

## 2. EVALUATION METRIC
- **Matchmaking**: Gaussian N(μ,σ2) Skill Rating system. Win increases μ, Loss decreases μ. Draw moves μ towards mean.
- **Leaderboard**: Best scoring agent of the two most recent active submissions is tracked on the leaderboard.

## 3. FEATURE HYPOTHESES
- None yet. Need to parse the Observation state space first.
