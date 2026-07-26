# 01_JOURNEY_LOG

**PROJECT:** Pokemon TCG Competition
**SYSTEM RULE:** APPEND ONLY. NEVER OVERWRITE PREVIOUS ENTRIES. STRICT TIMESTAMPS REQUIRED.

## ENTRY 001: Initialization & Environment Setup
**Timestamp:** 2026-07-26 14:25:00 +05:30
**Active Persona:** The Planner

**Hypothesis / Action:**
Initialized the Titan Universal system. Setting up the 4 core files in the workspace.

**Outcome / Observations:**
Successfully created 00_ENV_MANIFEST.md, 01_JOURNEY_LOG.md, 02_EXPERIMENT_TRACKER.md, and 03_META_RESEARCH.md. Workspace is currently empty except for git files.

**Next Steps:**
Awaiting user input for hardware target, constraints, and initial dependencies to update 00_ENV_MANIFEST.md.

## ENTRY 002: Environment Acknowledgment & Skill Update
**Timestamp:** 2026-07-26 14:28:30 +05:30
**Active Persona:** The Planner

**Hypothesis / Action:**
User provided preliminary hardware details (Local Machine, 8-core CPU, RTX 3050). User also insightfully noted that hardware constraints can only be finalized *after* analyzing the dataset and problem architecture. Updated the Titan Universal master skill to reflect this rule and logged the user's default hardware.

**Outcome / Observations:**
Updated `00_ENV_MANIFEST.md` with preliminary local hardware. Updated master skill directive.

**Next Steps:**
Transition to Phase 0 (Meta-Research) to understand the Pokemon TCG Kaggle competition problem statement, dataset size, and architecture requirements.

## ENTRY 003: Proactive Investigation & Meta-Research
**Timestamp:** 2026-07-26 14:31:00 +05:30
**Active Persona:** The Researcher

**Hypothesis / Action:**
User corrected the system's reliance on manual problem input. Updated the master directive to mandate proactive web searches. Searched the web for the "Pokemon TCG AI Battle Challenge" on Kaggle. Extracted the problem statement (building AI for the cabt Engine simulator) and evaluation criteria (win rate, stability, and strategy report). 

**Outcome / Observations:**
Successfully populated `03_META_RESEARCH.md`. The project is a Reinforcement Learning challenge dealing with Partially Observable Markov Decision Processes (POMDPs) rather than a standard tabular data task.

**Next Steps:**
Transition to Phase 1 (Data Ingestion & Memory Profiling). I need to acquire or locate the "cabt Engine" simulator SDK provided by Kaggle to profile the environment state arrays and memory requirements.

## ENTRY 004: Dataset Ingestion Attempt & Auth Failure
**Timestamp:** 2026-07-26 14:32:45 +05:30
**Active Persona:** The Data Engineer

**Hypothesis / Action:**
Attempted to download the Pokemon TCG Kaggle competition dataset autonomously using the Kaggle CLI (`kaggle competitions download -c pokemon-tcg-ai-battle-challenge-strategy`).

**Outcome / Observations:**
The `input` directory was created, but the Kaggle API call failed due to missing authentication credentials on the local host.

**Next Steps:**
Awaiting user to authenticate the Kaggle CLI on their machine before re-attempting the download.

## ENTRY 005: Dataset Ingestion & Simulator Profiling
**Timestamp:** 2026-07-26 14:38:50 +05:30
**Active Persona:** The Researcher / The Data Engineer

**Hypothesis / Action:**
Downloaded and unzipped the Kaggle dataset (`pokemon-tcg-ai-battle-challenge-strategy`). Investigated the contents and found only card CSVs and PDFs. Conducted proactive research to locate the `cabt Engine` simulator.

**Outcome / Observations:**
The dataset contains `EN_Card_Data.csv` (359 KB) and PDFs. Memory footprint is trivial. More importantly, research indicates the `cabt Engine` backend is compiled as a Linux/AMD64 shared object (`libcg.so`). Since the primary host is Windows, this presents a significant platform compatibility issue. Local execution will likely require Docker, Windows Subsystem for Linux (WSL2), or shifting the primary host to Kaggle Notebooks. 

**Next Steps:**
Inform the user via The Explainer about the dataset contents and the critical platform compatibility roadblock. We need to decide on WSL2 vs Docker vs Kaggle Notebooks before writing the first agent script.

## ENTRY 006: Planning Mode & Implementation Blueprint
**Timestamp:** 2026-07-26 14:39:50 +05:30
**Active Persona:** The Planner

**Hypothesis / Action:**
User pointed out that we leaped into Phase 1 without establishing a high-level roadmap and technical plan. Transitioned back to The Planner to formalize the architecture.

**Outcome / Observations:**
Created `implementation_plan.md` outlining the use of Proximal Policy Optimization (PPO), a custom Gymnasium wrapper for the `cabt Engine`, state vectorization strategies, and a 5-step Titan Phase map.

**Next Steps:**
Awaiting user approval of the implementation plan, specifically regarding the algorithm choice (PPO) and platform choice (WSL2).

## ENTRY 007: Implementation Plan Revision (BC -> PPO)
**Timestamp:** 2026-07-26 14:42:20 +05:30
**Active Persona:** The Planner

**Hypothesis / Action:**
User asked to rethink the implementation plan based on everything we learned, specifically taking into account the Ground-Truth Anchoring Rule. Searched Kaggle data and discovered that Kaggle provides top-rated episode replays daily. 

**Outcome / Observations:**
This massively shifted our architectural plan. Rewrote `implementation_plan.md` to start with Behavior Cloning (Imitation Learning) to pre-train the agent on Kaggle's top replays, drastically accelerating the RL training process before fine-tuning with PPO. Confirmed WSL2 as the optimal execution environment.

**Next Steps:**
Awaiting user approval on the revised implementation plan artifact.

## ENTRY 008: Ground-Truth Validation & Plan Reversion
**Timestamp:** 2026-07-26 14:46:10 +05:30
**Active Persona:** The Planner

**Hypothesis / Action:**
User demanded that we pull the ENTIRE RAW TEXT, not just a search snippet. Wrote a custom Playwright scraper (`scratch/scrape.py`) to bypass Kaggle's SPA javascript and dump the raw text of the Overview and Data tabs.

**Outcome / Observations:**
The ground-truth text contained NO MENTION of "Replay Data" or "Behavior Cloning". The web search tool had hallucinated or conflated that information from a different competition. The user's Ground-Truth Anchoring rule literally just saved the project from chasing non-existent dataset features. Reverted the `implementation_plan.md` back to pure PPO Self-Play.

**Next Steps:**
Awaiting user approval on the restored PPO implementation plan and WSL2 platform execution.

## ENTRY 009: URL Correction & Replay Validation
**Timestamp:** 2026-07-26 14:51:30 +05:30
**Active Persona:** The Planner

**Hypothesis / Action:**
User pointed out that we scraped the Strategy competition URL instead of the Simulation competition URL. Updated the Titan Master Skill to explicitly require embedding the scraped URLs in `03_META_RESEARCH.md` for auditing. Queried the Kaggle CLI for the correct slug (`pokemon-tcg-ai-battle`) and ran the Playwright scraper again.

**Outcome / Observations:**
The Simulation competition Data tab **EXPLICITLY confirms** the existence of Replays: "we will enable a daily episode export of the top rated episodes (to help BC/RL/IL)." The dataset also contains 60 files including the `ptcg_engine` source code, which was missing from the Strategy download. Behavior Cloning is fully validated. Restored the `implementation_plan.md` to the highly efficient BC -> PPO two-stage architecture.

**Next Steps:**
Awaiting user approval on the finalized Implementation Plan artifact. Once approved, Phase 1 will begin by re-downloading the correct dataset using the Kaggle CLI.

## ENTRY 013: Native Windows Engine Discovery!
**Timestamp:** 2026-07-26 15:00:30 +05:30
**Active Persona:** The Data Engineer

**Hypothesis / Action:**
With the implementation plan approved, I began Phase 1. I successfully downloaded the 327MB Simulation dataset (`pokemon-tcg-ai-battle.zip`) via the Kaggle CLI and extracted it. While preparing to set up the WSL2 environment as planned, I inspected the `sample_submission/cg` folder.

**Outcome / Observations:**
MASSIVE DISCOVERY. Kaggle actually provided pre-compiled binaries for all operating systems! The folder contains `cg.dll` (Windows), `libcg.dylib` (Mac), and `libcg.so` (Linux). I inspected `sim.py` and confirmed that `ctypes` automatically loads `cg.dll` if the OS is Windows. **This means we DO NOT need WSL2.** We can run the Kaggle simulation engine natively on Windows, taking direct advantage of the RTX 3050 without any virtualization overhead.

**Next Steps:**
Move on to the final step of Phase 1: Scripting the download of the Kaggle Replay Data to prepare for Phase 3 (Behavior Cloning).

## ENTRY 014: Replay Dataset Acquired
**Timestamp:** 2026-07-26 15:01:30 +05:30
**Active Persona:** The Data Engineer

**Hypothesis / Action:**
To complete Phase 1, I searched the Kaggle API for the promised Replay Data using `kaggle datasets list -s "pokemon tcg"`. I found that Kaggle publishes daily episode datasets under the `kaggle` account (e.g., `kaggle/pokemon-tcg-ai-battle-episodes-2026-07-12`).

**Outcome / Observations:**
The datasets exist and are massive (approx. ~750MB per day). I initiated the download of a recent daily export into the `input/replays` directory. This data will serve as the ground truth for our Behavior Cloning (Phase 3). Phase 1 (Data Ingestion) is officially complete.

**Next Steps:**
Transition to Phase 2 (Environment Wrapping). I need to write a Python script that integrates with `cg.dll` and wraps it in a standard OpenAI `Gymnasium` environment (`src/env/tcg_env_wrapper.py`).

## ENTRY 015: PAUSE - Algorithmic Justification
**Timestamp:** 2026-07-26 15:03:00 +05:30
**Active Persona:** The Researcher

**Hypothesis / Action:**
User halted the transition to Phase 2, questioning the foundational choice of PPO and demanding proper research and explanation before writing any code. I had skipped the deep dive in Phase 0. 

**Outcome / Observations:**
Paused all execution. Wrote a detailed mathematical and domain-specific justification in `03_META_RESEARCH.md` comparing PPO to DQN (which fails on POMDPs) and AlphaZero (which fails on high stochasticity). 

**Next Steps:**
Awaiting user feedback and explicit approval on the theoretical choice of PPO before returning to Phase 2 environment wrapping.

## ENTRY 016: REWIND TO PHASE 0 - Ground Truth Analysis
**Timestamp:** 2026-07-26 15:05:00 +05:30
**Active Persona:** The Researcher

**Hypothesis / Action:**
The user rejected the algorithmic justification because it wasn't anchored in a deep analysis of the actual Kaggle competition (the problem statement). They demanded I update the skill and "go from the start". 
Updated `SKILL.md` to mandate presenting competition research before algorithms. I then completely rewrote `03_META_RESEARCH.md` to analyze the `00_GROUND_TRUTH.md` document, extracting the evaluation system, API constraints, and mechanical challenges of the Pokémon TCG simulator.

**Outcome / Observations:**
The `cabt Engine` handles rule enforcement automatically, returning a list of legal options each turn. Evaluation uses a Gaussian Skill Rating $N(\mu, \sigma^2)$, matching agents dynamically. This confirms our neural network only needs to learn policy selection over pre-filtered valid actions, rather than learning the game rules from scratch.

**Next Steps:**
Present this bespoke competition research to the user via The Explainer persona to prove I understand the problem space before we ever talk about PPO or code again.

## ENTRY 017: Executing Online Research Round
**Timestamp:** 2026-07-26 15:09:00 +05:30
**Active Persona:** The Researcher

**Hypothesis / Action:**
The user pointed out that summarizing the rules isn't true research. I completely skipped analyzing the community meta, public code notebooks, and Kaggle discussion boards to see what is actually working for other competitors. 
Updated `SKILL.md` to STRICTLY MANDATE an "ONLINE Research round" involving discussions and notebooks. 

**Outcome / Observations:**
Running Kaggle CLI (`kaggle kernels list -c pokemon-tcg-ai-battle`) and web searches to mine the Kaggle discussion forums for the current meta, common pitfalls, and community baselines.

**Next Steps:**
Compile the findings from the online research into `03_META_RESEARCH.md` and present the community meta to the user.

## ENTRY 018: Online Research Completed
**Timestamp:** 2026-07-26 15:10:00 +05:30
**Active Persona:** The Researcher

**Hypothesis / Action:**
Analyzed the output from the Kaggle CLI and Google Search tools regarding the competition forums and code notebooks.

**Outcome / Observations:**
Discovered that Rule-Based/Heuristic bots built around specific deck archetypes (Alakazam, Starmie, Mega Lucario) are currently dominating the public meta because out-of-the-box RL struggles with the action space. Found discussion posts proving that submitting more than 2 agents ruins the scheduling priority for matchmaking. Added this to `03_META_RESEARCH.md`.

**Next Steps:**
Present these concrete, scraped community findings to the user to prove that a thorough, external research round has been conducted before algorithm selection.

## ENTRY 019: Downloading Community Ground Truth
**Timestamp:** 2026-07-26 15:13:00 +05:30
**Active Persona:** The Researcher

**Hypothesis / Action:**
The user pointed out a massive flaw in my execution: I summarized the existence of top Kaggle notebooks but failed to actually download them. "Someone else has done the hard work FOR US! WHY SHOULD WE WASTE IT!?"
I updated `SKILL.md` to strictly mandate downloading/hoarding actual notebook files and discussion posts into a local directory during Phase 0.

**Outcome / Observations:**
Created `input/ground_truth/notebooks`. Currently running `kaggle kernels pull` to download the top RL/MCTS notebook and the top Rule-Based baseline notebooks directly to our local machine so we can dissect their code and reuse their heuristics/logic.

**Next Steps:**
Confirm the notebooks have successfully downloaded and inform the user.

## ENTRY 020: Documenting Code Research
**Timestamp:** 2026-07-26 15:14:00 +05:30
**Active Persona:** The Researcher

**Hypothesis / Action:**
The user pointed out that after downloading and reading the notebooks, I jumped straight to making an implementation plan without formally documenting the architectural discoveries in `03_META_RESEARCH.md`. I updated `SKILL.md` to strictly mandate this documentation step.

**Outcome / Observations:**
Added Section 1.6 to `03_META_RESEARCH.md`, formally documenting that Kiyota uses PyTorch `EmbeddingBag` and a Transformer Encoder-Decoder to handle the variable-length JSON states, and that Rozen uses UCB1 search trees on top of heuristics.

**Next Steps:**
Apologize to the user, present the updated research, and await their approval on the implementation plan.

## ENTRY 021: Finalizing the Architecture
**Timestamp:** 2026-07-26 15:18:00 +05:30
**Active Persona:** The Planner

**Hypothesis / Action:**
The user directed me to decide the answers to my own open questions in the implementation plan. 
Debate 1: BC vs Self-Play. Conclusion -> BC Pre-training is strictly necessary because the action space is too large for pure RL exploration from scratch.
Debate 2: Heuristic vs Self-Play. Conclusion -> We must use a Hybrid Pool of both past self-versions AND the downloaded heuristic bots so the RL agent learns how to beat the actual leaderboard meta.

**Outcome / Observations:**
Updated `implementation_plan.md` replacing the open questions with finalized Strategic Decisions.

**Next Steps:**
Request final user sign-off on the plan before moving to Phase 1 (Data Ingestion & Memory Profiling) to parse the 750MB Replay Dataset for BC training.

## ENTRY 022: Execution Phase Initiated
**Timestamp:** 2026-07-26 15:20:00 +05:30
**Active Persona:** The Builder

**Hypothesis / Action:**
The user approved the `implementation_plan.md` and explicitly commanded: "don't forget rigorous testing in every step.. go step by step. test and iterate! consider the objectives before u change the code to be logically proof!"
I created a `task.md` artifact breaking the execution down into 5 heavily tested steps (Transformer, BC Parser, BC Training, Env Wrapper, PPO).

**Outcome / Observations:**
Created project directories (`src/model`, `src/data`, `src/env`, `tests`). 

**Next Steps:**
Implement Step 1: Extract Kiyota's Transformer architecture into `src/model/transformer_policy.py` and write a unit test to verify tensor flow.

## ENTRY 023: Step 1 Complete (Transformer Architecture)
**Timestamp:** 2026-07-26 15:22:00 +05:30
**Active Persona:** The Builder

**Hypothesis / Action:**
Wrote a Python script to physically extract the Transformer Encoder/Decoder classes and the JSON-to-Tensor `SparseVector` mapping functions from Kiyota's notebook into our `src/model/transformer_policy.py`. 
Wrote `tests/test_transformer.py` which initializes the native Kaggle engine (`cg.dll`), pulls a raw JSON `Observation` from a dummy battle, and passes it through the Transformer.

**Outcome / Observations:**
SUCCESS. The `cg.dll` initialized natively on Windows (loading 1267 cards). The PyTorch Transformer successfully encoded the raw JSON state and outputted a valid Critic Value and a Policy Logit list corresponding to the exact number of legal moves.

**Next Steps:**
Inform the user of the successful test and proceed to Step 2 (Behavior Cloning Replay Parser) to convert the 750MB Kaggle dataset into training inputs.

## ENTRY 024: Step 2 Complete - The Dataset Pivot
**Timestamp:** 2026-07-26 15:29:00 +05:30
**Active Persona:** The Builder

**Hypothesis / Action:**
Started building the BC parser to read the 750MB of `.json` Kaggle replays. Discovered a fatal flaw: Kaggle replays strip the player's 60-card `deck` list for anti-cheating purposes. The PyTorch Transformer (`get_encoder_input`) requires the starting deck to initialize the state tensors. Therefore, we cannot Behavior Clone using the public replays.
**Pivot:** I wrote `src/data/generate_heuristics.py` to initialize the Kaggle engine locally, pass it a known deck, and force the top LB 950+ Rule-Based bot (Roman Rozen's script) to play itself. At every step, the script calculates the Transformer tensors and logs the heuristic bot's chosen action as the target label.

**Outcome / Observations:**
SUCCESS. The script generated 5 games (420 flawless state-action tensor pairs) in under 1 second and saved them to `input/bc_dataset.pkl`. We now have an infinitely scalable data engine to Behavior Clone the meta.

**Next Steps:**
Inform the user of this massive technical pivot, and proceed to Step 3 (BC Pre-Training Loop).

## ENTRY 025: Steps 3 & 4 Complete (Trainer and Environment)
**Timestamp:** 2026-07-26 15:34:00 +05:30
**Active Persona:** The Builder

**Hypothesis / Action:**
- **Step 3 (BC Trainer):** Wrote `src/train_bc.py` to load the dataset, initialize the Transformer on GPU, and run a Cross-Entropy training loop. 
- **Step 4 (Environment Wrapper):** Wrote `src/env/tcg_env.py` (`PokemonTCGEnv`) to abstract the Kaggle C++ `cg.dll` logic (ctypes, JSON parsing) into clean `reset()` and `step(action)` methods. Tested it against the rule-based bot.

**Outcome / Observations:**
SUCCESS. Step 3 verified that the PyTorch tensors flow perfectly through the Transformer without shape mismatches. Step 4 verified that the local Kaggle engine can be cleanly wrapped into a standard RL environment loop for PPO rollout collection.

**Next Steps:**
Inform the user, and begin the final and most complex step: Step 5 (The Hybrid PPO Trainer).

## ENTRY 026: Step 5 Complete (Hybrid PPO Engine Validated)
**Timestamp:** 2026-07-26 15:42:00 +05:30
**Active Persona:** The Builder

**Hypothesis / Action:**
- **Step 5 (Hybrid PPO):** Wrote `src/train_ppo.py` to test the final piece of the architecture. The script initializes the Transformer, runs 2 full episodes inside `PokemonTCGEnv` playing *against* the Heuristic Bot, collects the Trajectories, computes Generalized Advantage Estimation (GAE), and applies the PPO clipped loss objective.

**Outcome / Observations:**
SUCCESS. The script successfully played both episodes against the Heuristic Bot. It collected full state/action/reward trajectories and computed Advantage. The PPO Loss and Value Loss successfully updated the network weights without any `NaN` values or tensor mismatches. The core mathematical pipeline is complete.

**Next Steps:**
Create `walkthrough.md` to summarize the completed pipeline and present the results to the user.

## ENTRY 027: Production Pipeline Complete (Steps 6-8)
**Timestamp:** 2026-07-26 16:00:00 +05:30
**Active Persona:** The Builder

**Hypothesis / Action:**
Implemented the three production MLOps scripts:
- **Step 6:** `src/train_ppo_scale.py` - Scaled PPO loop with 50/50 opponent sampling (Heuristic Bot vs Past Checkpoints), periodic checkpointing, colorful ANSI terminal logging with threshold-based colors, and entropy tracking.
- **Step 7:** `src/eval/evaluate_elo.py` - Local tournament harness that loads a specific checkpoint via `--ckpt` and plays N matches against the baseline bot.
- **Step 8:** `src/package_submission.py` - One-click build script that bundles model weights, transformer_policy.py, and the Kaggle `main.py` hook into `submission.tar.gz`.

**Outcome / Observations:**
All three scripts executed successfully. The training loop achieves ~2.5 games/second. The packager correctly compressed the submission to `submission.tar.gz`. User began first 100,000 episode training run.

**Next Steps:**
Monitor training metrics and iterate on reward shaping.

## ENTRY 028: TrueSkill Expected Score (Evaluation Metric Fix)
**Timestamp:** 2026-07-26 16:22:00 +05:30
**Active Persona:** The Researcher

**Hypothesis / Action:**
The user identified a critical flaw in the evaluation metric. `evaluate_elo.py` was reporting raw Win Rate = `Wins / Total`, which treated Draws (72 out of 100 matches) identically to Losses. The user correctly argued that surviving 500 turns against a top-tier heuristic bot without losing is NOT equivalent to getting knocked out on Turn 2.

**Outcome / Observations:**
Research confirmed Kaggle uses a Gaussian Skill Rating system where Draws count as 0.5 points (like Chess ELO). Patched `evaluate_elo.py` to output `TrueSkill Expected Score = (Wins + 0.5 * Draws) / Total`. The Episode 500 model's score went from a misleading 11% Win Rate to a much more accurate 47% Expected Score. Logged in `04_USER_UNDERSTANDING.md` Entry 019.

**Next Steps:**
Investigate whether the same Draw/Truncation problem affects the training loop itself.

## ENTRY 029: CRITICAL BUG FIX - GAE Truncation Bootstrap
**Timestamp:** 2026-07-26 16:27:00 +05:30
**Active Persona:** The Model Architect

**Hypothesis / Action:**
The user asked: "Is that gonna be a problem with training too?" This triggered a deep investigation. Web research confirmed a well-known RL pitfall: when an episode is TRUNCATED (hit the 500-step limit), the GAE function should NOT treat the next-state value as 0.0 (terminal). Instead, it must BOOTSTRAP from the Critic's value estimate of the current state, because the game isn't actually over -- we just stopped it early.

Our `compute_gae()` was hardcoding `values + [0.0]` for ALL episodes, meaning truncated episodes were teaching the agent: "states near the 500-step limit are worth nothing." This could cause the agent to learn a degenerate stalling strategy (survive until truncation for a "safe" 0.0 rather than risk attacking for +1.0).

**Outcome / Observations:**
Fixed `compute_gae()` to accept a `bootstrap_value` parameter. When `truncated=True`, the Critic evaluates the final state and passes its estimate as the bootstrap. When terminated naturally (Win/Loss), bootstrap remains 0.0. Also added explicit W/D/L tracking counters.

**Next Steps:**
User should restart the training run with the fixed script. The agent should now learn to be more aggressive and convert Draws into Wins.
