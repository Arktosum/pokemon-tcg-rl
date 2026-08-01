# 01_JOURNEY_LOG
**SYSTEM RULE:** APPEND ONLY. 

## ENTRY 001: TITAN Activation and Phase 1 Initialization
**Timestamp:** 2026-07-30 12:32:28 +05:30
**Hypothesis / Action:** Activating TITAN V5.0. User approved Phase 1. Initializing core files and setting up local simulation environment to test the baseline random agent.
**Outcome / Observations:** N/A (In Progress)
**Next Steps:** Run local simulation with `kaggle-environments` and generate a replay file.
 
**Update 12:37:** Step 1.1 completed successfully. Local simulation ran using kaggle-environments cabt engine. Generated replay.json successfully.
  
**Update 12:44:** Re-configured run_local_battle.py to output replay files with datetime stamps (e.g., replay_20260730_124400.json). Added strict logging constraints to 00_DIRECTIVES.md per user request.
  
**Update 13:12:** User manually added the replay configuration dump to CABT_ENGINE_API_DOCUMENTATION.md and made minor tweaks to run_local_battle.py.

## ENTRY 002: Behavioral Cloning Dataset Extraction
**Timestamp:** 2026-07-31 19:35:00 +05:30
**Hypothesis / Action:** Executing Step 1 of Phase 2.1. Wrote `prepare_bc_dataset.py` to parse JSON replays into PyTorch tensor tuples. Wrote `test_prepare_dataset.py` with pytest to rigorously verify tensor mathematical shapes and edge cases.
**Outcome / Observations:** `pytest` passed (3 tests: valid replay, draw replay, corrupted replay). Ran benchmark on 50 files (took ~20 seconds). Extracted 9,021 valid training samples. Extrapolating to the full 4,385 dataset, this yields ~790,000 highly structured expert-play samples.
**Next Steps:** Step 2: Implement the `TitanTransformer` architecture (`model.py`) to consume these extracted tensors.

## ENTRY 003: Neural Network Architecture & Human-Readable Verification
**Timestamp:** 2026-07-31 19:56:00 +05:30
**Hypothesis / Action:** Executing Step 2 of Phase 2.1. Developed the TitanTransformer architecture with a TransformerEncoder and dynamic Pointer Network action masking. Per the user's /learn rule, wrote a custom `conftest.py` plugin to enforce a "Human-Readable Test Summary" format with Unicode icons and embedded docstring descriptions for rigorous testing.
**Outcome / Observations:** Architecture tests successfully executed via CMD to circumvent PowerShell pipeline errors. Log files generated successfully. The model elegantly handles dynamic batched action spaces using `-inf` masking.
**Next Steps:** Step 3: Implement the Training Loop (`train_bc.py`) using the Behavioral Cloning dataset to warm-start the model weights.

## ENTRY 004: [CRITICAL BUG] Behavioral Cloning CUDA Crash
**Timestamp:** 2026-07-31 20:05:00 +05:30
**Hypothesis / Action:** Executed `train_bc.py` for a 1024-sample Proof of Concept run. 
**Outcome / Observations:** The first batch returned `Loss: inf` and then immediately crashed with a `CUDA error: device-side assert triggered` originating from `Loss.cu:250 (t >= 0 && t < n_classes)`. This indicates that a target action index in the dataset is either exceeding `max_N` (out of bounds) or pointing to an invalid masked `-inf` action.
**Next Steps:** CIRUIT BREAKER TRIGGERED. Yielding to User for manual intervention to investigate the Kaggle engine's `action` index logic versus `legal_option_count`.
- [PASSED] train_bc.py POC successfully executed 10,000 samples after rigorous bounds filtering, fully resolving the CUDA out-of-bounds assert. The behavioral cloning warm-start logic is officially stable. 
  
## ENTRY 004: Phase 2.1 Behavioral Cloning - COMPLETE  
**Timestamp:** 2026-07-31 21:45:26 IST  
**Hypothesis / Action:** Run full 10-epoch BC training on 340,609 sanitized samples with 90/10 train/val split, model checkpointing on val loss improvement, and timestamped rich logging.  
**Outcome / Observations:**  
- ZERO CUDA crashes. Dataset fully traversed across all 10 epochs.  
- Best Val Loss: 1.4073 at Epoch 9.  
- Best checkpoint saved: 20260731_214125_titan_bc.pt  
- Train Loss converged: 1.77 (Ep1) -> 1.39 (Ep10). Val Loss stable 1.40-1.42 (no overfitting).  
**Next Steps:** Begin Phase 2.2 - PPO training loop with env_wrapper.py and custom reward shaping. 

## ENTRY 005: PPO Dashboard Refactor & Daemon Training Launch
**Timestamp:** 2026-08-01 11:29:40 +05:30
**Hypothesis / Action:** Executed user /goal to refactor train_ppo.py with comprehensive terminal dashboards, robust checkpointing, ETA tracking, and flat CSV output. Ran pytest unit tests on the logging/checkpointing fallback logic which successfully passed. Launched PPO training against RandomBot as a background daemon and initiated a 5-minute cron schedule to monitor the metrics.
**Outcome / Observations:** N/A (Daemon just started. Monitoring loop initiated.)
**Next Steps:** Parse metrics.csv when the cron triggers. Report the PPO Win Rate and KL divergence. If Win Rate > 80%, terminate training, release lock, and swap to SetupBot.
- [2026-08-01] Refactored PPO training to use a synchronous Multiprocessing VectorEnv (16 workers), dropping iteration time from ~80s to ~5s. Implemented dynamic Tensor collation for batched inputs to TitanTransformer. Verified with Pytest. Re-launched daemon. 
