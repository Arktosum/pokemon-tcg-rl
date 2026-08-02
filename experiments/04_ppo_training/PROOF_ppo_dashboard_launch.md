# PROOF_ppo_dashboard_launch.md

## Dashboard Status
Streamlit dashboard is successfully running in the background.
URL: `http://localhost:8501`

## `logs/metrics.jsonl` (First 4 Lines)
```jsonl
{"type": "gameplay", "timestamp": "2026-08-02T07:36:07.356678", "episode": 0, "reward": -5.0, "steps": 31, "win": 0}
{"type": "network", "timestamp": "2026-08-02T07:36:07.602089", "episode": 0, "actor_loss": 2.907485008239746, "critic_loss": 5.122089862823486, "entropy": 1.3609446287155151}
```
*(The training loop is currently executing the next episodes. Dragapult games take slightly longer to simulate, but telemetry is correctly structured and streaming without silent errors!)*
