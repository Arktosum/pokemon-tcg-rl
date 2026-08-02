# 04_USER_UNDERSTANDING
**SYSTEM RULE:** APPEND ONLY. 

## ENTRY 001: Phase 1 Approval
**Timestamp:** 2026-07-30 12:32:28 +05:30
**User Feedback:** "okay. execute Step 1.1 and provide me with proper results and testing and show the output logs. also point me to the command so i can run and see it for myself. keep the organization of the repo clean."
**Adjustment Needed:** Ensure the repo remains clean. Will create a dedicated `experiments/01_baseline` directory for Step 1.1 execution. Must explicitly provide the run command to the user.
 

## ENTRY 003: Acknowledging Manual User Adjustments
**Timestamp:** 2026-07-30 13:12:00 +05:30
**User Feedback:** "ok.. cool i made a few changes to the script and also added the replay config to the CABT md file.. acknowledge the changes.."
**Adjustment Needed:** Noted the user manually tweaked `run_local_battle.py` and appended the baseline schema to `CABT_ENGINE_API_DOCUMENTATION.md`. System is perfectly in sync.
## ENTRY 003: PPO Bug Extermination
**Timestamp:** 2026-08-02 07:09:06 
**User Feedback:** The user strictly commanded NEVER to use generic try/except blocks to silently suppress errors in the rollout buffer. Crashing is better than feeding dummy tensors into the model.
**Adjustment Needed:** Adhere to absolute transparency. Let PyTorch throw tracebacks to expose the true root cause.
