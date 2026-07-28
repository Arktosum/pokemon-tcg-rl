# 00_DIRECTIVES

## IMMUTABLE GOALS & CONSTRAINTS

* **Core Mandate:** Build an AI agent that plays the Pokémon Trading Card Game (PTCG AI Battle Challenge Simulation), aiming for a leaderboard rating of 1100+. Build a hybrid agent using a PUCT tree search combined with a Neural Network for Policy/Value estimation, trained via Self-Play and Behavioral Cloning.

* **CRITICAL CONTEXT CORRECTION (Phase 46):** The "score" returned by a submission initially (e.g., 600.0) is not a quality metric. It is purely the result of a Validation Episode (often self-play or against a dummy) to check if the bot runs without crashing. The actual competitive ranking is determined by wins/losses in the matchmaking pool over time. Validation scores measure crash-resistance, not model skill.
* **CLI Commands:** `kaggle competitions download -c pokemon-tcg-ai-battle`

* **Paths & Constants:** Dataset stored in `data/`

Note (Phase 7): Acknowledged the 5-submission-per-day Kaggle constraint and the necessity of strict local validation against Greedy Agents before submission.

Note (Phase 7): Acknowledged the 5-submission-per-day Kaggle constraint and the necessity of strict local validation against Greedy Agents before submission.

