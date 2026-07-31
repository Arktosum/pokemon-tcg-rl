# 00_DIRECTIVES

## IMMUTABLE GOALS & CONSTRAINTS
* **Core Mandate:** Build an AI Training Agent to play the Pokémon Trading Card Game for the Kaggle Simulation Competition.
* **Logging Requirement:** ALWAYS save logs, replay files, and model artifacts with a precise YYYYMMDD_HHMMSS datetime stamp. Never overwrite past outputs.
* **CLI Commands:** `kaggle competitions submit -c pokemon-tcg-ai-battle ...`
* **Paths & Constants:** 
    * Simulator Engine: `cabt`
    * Max Submission Size: 197.7 MiB
    * Max Runtime per Step: 3 seconds
    * Submission Format: `submission.tar.gz` with `main.py` at the root and `deck.csv`.
