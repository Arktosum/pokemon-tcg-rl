# TITAN SYSTEM DIRECTIVE: SELF-PLAY CURRICULUM & CHECKPOINT ARCHITECTURE - PROOF

## Configuration Changes
- Added `curriculum` to `config.yaml`.
- Updated `ppo_config.py` to parse `CurriculumConfig`.

## New Modules
- `ppo_checkpoint.py` ensures robust weight persistence using standard PyTorch `state_dict` conventions.
- `ppo_bridge.py` generates a self-contained agent script for the Kaggle engine using PyTorch model.
- `ppo_sampler.py` orchestrates the curriculum based on YAML probabilities.

## Updates
- Wired it to the orchestrator (`train_sequential.py`) to properly load checkpoints and sample opponents dynamically, explicitly forcing the self-play opponent in `--test_mode`.

## Logs
```
[kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO: Successfully loaded OpenSpiel environments: 39.
[kaggle_environments.envs.open_spiel_env.open_spiel_env] INFO: OpenSpiel games skipped: 2.
Episode 0 | PG Loss: 0.0004 | V Loss: 0.0584 | Entropy: 1.1037
Validation Win Rate: 1.00
Episode 1 | PG Loss: -0.0004 | V Loss: 0.0219 | Entropy: 1.2097
Validation Win Rate: 0.00
```
