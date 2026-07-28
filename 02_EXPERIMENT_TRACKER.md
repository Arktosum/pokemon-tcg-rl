# 02_EXPERIMENT_TRACKER

**SYSTEM RULE:** APPEND ONLY. 

## EXPERIMENT LOG

| Exp ID | Timestamp | Model / Strategy | CV Score | Notes | Lock Status |

| :--- | :--- | :--- | :--- | :--- | :--- |

| `001` | 2026-07-28 12:05 | V1 State Vectorization & Env Wrapper | N/A | Testing gym wrapper | [RELEASED] |

| `002` | 2026-07-28 12:09 | V2 Evidence-Backed Vectorization | N/A | Added Card ID embeddings and Handcrafted stats | [RELEASED] |

| `003` | 2026-07-28 12:15 | Phase 3: Dual-Head NN & PUCT | N/A | Two-stream AlphaNet & MCTS Node Logic | [RELEASED] |

| `004` | 2026-07-28 12:18 | Phase 4: Self-Play Training Loop | N/A | Self-Play Data Generation & Loss Engineering | [RELEASED] |

| `005` | 2026-07-28 12:26 | Phase 5: Behavioral Cloning Pre-training | N/A | Supervised bootstrapping on Kaggle JSON replays | [ACTIVE LOCK] |

| `006` | 2026-07-28 12:28 | Phase 6: Evaluation & Kaggle Packaging | N/A | Win Rate & TarGz | [ACTIVE LOCK] |

| `007` | 2026-07-28 12:38 | Phase 7: Scale-Up & Arena Eval | N/A | Massive BC and Greedy Agent Evaluation | [ACTIVE LOCK] |



| `008` | 2026-07-28 12:43 | Phase 8: Self-Play RL & Repaired Arena | N/A | 250 Self-Play games via AlphaZero loop | [ACTIVE LOCK] |


| `009` | 2026-07-28 12:46 | Phase 9: Forensic Diagnostic | N/A | Loss Autopsy, Tensor Audit, Reward Audit | [ACTIVE LOCK] |

| `010` | 2026-07-28 12:56 | Phase 10: Bug Patching & Final Validation | N/A | Patched env.py and re-ran full RL pipeline | [ACTIVE LOCK] |

| `011` | 2026-07-28 13:00 | Phase 11: Deep RL Scale-Up | N/A | Scaled MCTS to 50, c_puct to 1.5, BC to 30, SP to 1000 | [ACTIVE LOCK] |

| `012` | 2026-07-28 13:10 | Phase 12: True MCTS Engine Rewrite | N/A | Rewrote puct.py for real MCTS; identified global C-state bug | [ACTIVE LOCK] |

| `013` | 2026-07-28 13:12 | Phase 13: Architecture Pivot to PPO | N/A | Abandoned MCTS, Built Actor-Critic PPO | [ACTIVE LOCK] |

| `014` | 2026-07-28 13:25 | Phase 14: Workspace Cleanup & PPO Stabilization | N/A | Cleaned workspace into src/, patched NaN bug & type casts | [ACTIVE LOCK] |

| `015` | 2026-07-28 13:38 | Phase 15: PPO Entropy Tuning | N/A | Modified entropy masking, lr=3e-4, 2000 episodes | [ACTIVE LOCK] |

| `016` | 2026-07-28 13:46 | Phase 16: Dense Reward Shaping | N/A | Added +/- 0.1 per prize card taken, 2500 episodes | [ACTIVE LOCK] |

| `017` | 2026-07-28 13:54 | Phase 17: Residual Network Scale-Up | N/A | 4-layer ResNet-256, 30 epoch BC, 2000 episode PPO | [ACTIVE LOCK] |

| 018 | Phase 18: Curriculum Learning & Noise Training | Curriculum Rewards (+0.05 bench/energy) + 30% Random opponent | [ACTIVE LOCK] |

| Phase 18 | ResNet-256 + PPO + Curriculum (3000 ep) | 76.6% (383/500) | 52.4% (262/500) | FAILED (Under 95/80 target) |

| Exp ID 019 | Phase 19: Policy Distillation & Aggression Bias | TBD | TBD | [ACTIVE LOCK] |

| Exp ID 020 | Phase 20: League Training & Mask Reversion | TBD | TBD | [ACTIVE LOCK] |

| Exp ID 021 | Phase 21: Kaggle Parity & Transformer | TBD | TBD | [ACTIVE LOCK] |

| Exp ID 022 | Phase 22: BC Value Bug & Transformer Warmup | TBD | TBD | [ACTIVE LOCK] |

| Exp ID 023 | Phase 23: Real Data & Elo Gating | TBD | TBD | [ACTIVE LOCK] |

| Exp ID 024 | Phase 24: Hallucination Audit & Proof of Life | TBD | TBD | [ACTIVE LOCK] |

| Exp ID 025 | Phase 25: Workspace Quarantine & True BC | TBD | TBD | [ACTIVE LOCK] |

| Exp ID 026 | Phase 26: Massive Scale-Up & League Training | TBD | TBD | [ACTIVE LOCK] |

| Exp ID 027 | Phase 27: Kaggle Deployment | TBD | TBD | [ACTIVE LOCK] |

| Exp ID 028 | Phase 28: Kaggle Environment Forensics | TBD | TBD | [ACTIVE LOCK] |

| Exp ID 029 | Phase 29: Root Cause Verification | TBD | TBD | [ACTIVE LOCK] |
