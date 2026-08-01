# 02_EXPERIMENT_TRACKER
**SYSTEM RULE:** APPEND ONLY. 

## EXPERIMENT LOG
| Exp ID | Timestamp | Model / Strategy | CV Score | Notes | Lock Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `002` | 2026-07-31 19:24 | TitanTransformer (BC + PPO) | N/A | Warm-starting with 4.3k replays, custom masked PPO | [ACTIVE LOCK] |
| `003` | 2026-07-31 21:45 | TitanTransformer BC Warmstart (10 epochs, 340,609 samples) | Val Loss: 1.4073 | 90/10 split, checkpoint at Ep9, ~4min/epoch RTX, no overfitting. Weights: 20260731_214125_titan_bc.pt | [RELEASED - SUCCESS] |
| `004` | 2026-08-01 10:41 | Rule-Based Bot Curriculum & PPO Wrapper | N/A | Implemented GreedyBot (Aggressive priority) & TacticalBot (Retreat threshold). Built Gym wrapper + Rollout Buffer. PPO smoke tests passing. | [RELEASED - SUCCESS] |
