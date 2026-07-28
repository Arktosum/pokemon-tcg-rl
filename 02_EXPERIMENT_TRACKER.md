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

| Exp ID 019 | Phase 19: Architecture Pivot (PPO to Transformer) | N/A | Investigated NaN bug; decided to pivot to Transformer architecture | [RELEASED] |
| Exp ID 020 | Phase 20: Transformer Setup | N/A | Implemented custom Transformer for state processing | [RELEASED] |
| Exp ID 021 | Phase 21: Kaggle Parity | N/A | Addressed PyTorch inference parity on CPU | [RELEASED] |
| Exp ID 022 | Phase 22: Value Bug Patch | N/A | Patched value bug and warmed up Transformer | [RELEASED] |
| Exp ID 023 | Phase 23: Real Data Fetch | N/A | Real dataset fetched and integrated into training | [RELEASED] |
| Exp ID 024 | Phase 24: Hallucination Audit | N/A | Addressed hallucinated paths and validated proof of life | [RELEASED] |
| Exp ID 025 | Phase 25: True BC Pretraining | N/A | Cleaned environment and pre-trained BC accurately | [RELEASED] |
| Exp ID 026 | Phase 26: Massive Scale-Up | N/A | Scaled up League Training with Transformer | [RELEASED] |
| Exp ID 027 | Phase 27: Kaggle Deployment | N/A | Created submission bundle | [RELEASED] |
| Exp ID 028 | Phase 28: C++ Engine Forensics | N/A | Diagnosed Kaggle simulation C++ execution bugs | [RELEASED] |
| Exp ID 029 | Phase 29: Root Cause Verification | N/A | Verified local vs remote environment mismatches | [RELEASED] |
| Exp ID 030 | Phase 30: ONNX Pivot | N/A | Decided to export model to ONNX to bypass Kaggle PyTorch limits | [RELEASED] |
| Exp ID 031 | Phase 31: ONNX Parity | N/A | Validated ONNX inference numerical parity | [RELEASED] |
| Exp ID 032 | Phase 32: Fallback Logic Implementation | N/A | Added try-except logic for robust Kaggle deployment | [RELEASED] |
| Exp ID 033 | Phase 33: True Deployment | N/A | Shipped ONNX wheel with submission tarball | [RELEASED] |
| Exp ID 034 | Phase 34: Telemetry (Abandoned) | N/A | Skipped due to step 0 global crash | [RELEASED] |
| Exp ID 035 | Phase 35: Forensics & Repair | N/A | Investigated the fatal error occurring before agent loop | [RELEASED] |
| Exp ID 036 | Phase 36: Silent Failure Trap | ERROR | Validated cross-platform binaries but hit silent failure | [RELEASED] |
| Exp ID 037 | Phase 37: Fallback Shape Patch | ERROR | Fixed fallback nested list format to flat list | [RELEASED] |
| Exp ID 038 | Phase 38: Step 0 Deck Injection | ERROR | Discovered Kaggle expects full 60-card deck on Step 0 | [RELEASED] |
| Exp ID 039 | Phase 39: Absolute Encapsulation | ERROR | Moved `__file__` inside agent scope to avoid Kaggle string-exec bugs | [RELEASED] |
| Exp ID 040 | Phase 40: Unbreakable Shell | ERROR | Wrapped `agent()` internals in master `try...except` block | [RELEASED] |
| Exp ID 041 | Phase 41: Vanilla Baseline | COMPLETE (433.9) | Removed ONNX entirely, fed sample integer deck on Step 0 | [RELEASED] |
| Exp ID 042 | Phase 42: ONNX Brain Transplant | COMPLETE (345.6) | Injected Top-K ONNX masking logic + Dummy Model fallback check | [ACTIVE LOCK] |

| Exp ID 043 | 2026-07-28 22:30 | Baseline Eval: TITAN_TRANSFORMER_LEAGUE_01.pt vs GreedyAgent | 28.3% WR (283/1000) | Confirmed massive regression from self-reported Phase 18 stats | [RELEASED] |

| Exp ID 044 | 2026-07-28 22:41 | Greedy-Targeted PPO (250 ep): TITAN_GREEDY_PPO_01.pt | 46.5% WR (465/1000) | +18.2pp improvement over base. Entropy oscillated 0.09-0.90. Saved to checkpoints/. | [RELEASED] |

| Exp ID 045 | 2026-07-28 23:30 | BC Strategy Decision: Top-10 Elo Replay Scraping | N/A | Pivot from PPO to BC from 1130+ Elo games. Filter: MIN_SUB_SCORE=1130. 6,479 clean episodes identified. | [ACTIVE LOCK] |

| Exp ID 046 | 2026-07-29 00:08 | Top-Elo Replay Download: 9 teams, 6,479 episodes | IN PROGRESS | Sequential download at 1s/replay. ETA ~1hr 48min. Target model: TOP_ELO_BC_MODEL. | [ACTIVE LOCK] |

