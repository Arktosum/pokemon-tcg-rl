# Full-Spectrum Telemetry Proof

Successfully ran `train_sequential.py --test_mode`. Below is a sample JSON line demonstrating that all extended telemetry fields (sps, kl_divergence, explained_variance, grad_norm, opponent, duration, etc.) are successfully being logged.

```json
{"timestamp": "2026-08-02T18:20:02.164476", "episode": 1, "opponent": "random", "steps": 44, "duration_sec": 0.869, "sps": 50.63, "reward": 1.0, "win": 1, "pg_loss": 0.0012960184831172228, "v_loss": 0.22357158362865448, "entropy": 1.5648328065872192, "kl_divergence": 7.408488454530016e-05, "explained_variance": 0.12029439210891724, "grad_norm": 15.179338455200195, "val_win_rate": 0.0}
```
