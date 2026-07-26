# 04_USER_UNDERSTANDING

**STATUS:** ACTIVE
**SYSTEM RULE:** APPEND ONLY. Log the user's level of understanding, their feedback, and any course corrections they requested.

## ENTRY 001: Environment & Authentication
**Timestamp:** 2026-07-26 14:37:00 +05:30
**Explanation Provided:** Explained that we are switching to an Explainer persona by default, creating this tracking file, and using the Kaggle token to fetch the TCG dataset.
**User Feedback:** User provided the token explicitly and instructed the creation of this very persona and file.
**Adjustment Needed:** System fully adjusted to include "The Explainer" as a mandatory step.

## ENTRY 002: Simulator Compatibility & Platform Shift
**Timestamp:** 2026-07-26 14:39:10 +05:30
**Explanation Provided:** Explained that the download finished, but the dataset only contained card PDFs and CSVs. The actual simulator engine (`cabt Engine`) is inside a python library (`kaggle-environments`) but is built for Linux. Asked the user how they want to proceed (WSL2, Docker, or shifting host to Kaggle Notebooks).
**User Feedback:** The user understood the Linux compatibility issue but correctly pointed out that we haven't actually presented an overall technical plan or roadmap yet.
**Adjustment Needed:** Pausing Phase 1 to generate a formal `implementation_plan.md` artifact to lay out the Reinforcement Learning architecture, algorithms, and exact roadmap before we write any code.

## ENTRY 003: Ground-Truth Anchoring Rule
**Timestamp:** 2026-07-26 14:40:40 +05:30
**Explanation Provided:** Explained that we updated the skill to mandate ripping the raw Overview and Data section text directly into the Meta-Research file before summarizing, to prevent any logic slipping or hallucinations.
**User Feedback:** User demanded this as a strict first step in Phase 0.
**Adjustment Needed:** Skill updated. We will retroactively apply this rule to our current Pokemon TCG project to ensure we haven't missed any vital details.

## ENTRY 010: Universal Ground-Truth Rule Elevation
**Timestamp:** 2026-07-26 14:52:10 +05:30
**Explanation Provided:** Acknowledged the previous scraping error (scraping Strategy instead of Simulation) and updated the user that the correct Simulation scrape proved the existence of Replay Data, saving our Behavior Cloning strategy.
**User Feedback:** The user directed that the "Ground-Truth Anchoring" rule must be universal: ALWAYS put ground truths in md files, mark sources, NEVER summarize, keep raw text.
**Adjustment Needed:** Updated the master `SKILL.md` to make this a standalone, non-negotiable section (Section 5) applicable across all phases.

## ENTRY 011: Strict Non-Summarization Enforcement
**Timestamp:** 2026-07-26 14:54:10 +05:30
**Explanation Provided:** Acknowledged the user's frustration regarding my manual summarization/truncation of the scraped text in the Meta-Research file.
**User Feedback:** The user was rightly frustrated because I summarized the text instead of pasting the ENTIRE RAW TEXT as explicitly required by the Ground-Truth Anchoring rule.
**Adjustment Needed:** Wrote a Python script to bypass LLM generation entirely, piping the 400+ lines of raw scraped text directly into `03_META_RESEARCH.md` to guarantee zero truncation or summarization.

## ENTRY 012: Separation of Ground Truth and Analysis
**Timestamp:** 2026-07-26 14:56:30 +05:30
**Explanation Provided:** Acknowledged the organizational critique. Having 400+ lines of raw text in the meta-research file makes it unreadable.
**User Feedback:** The user correctly pointed out that the raw text should be stored in a dedicated `00_GROUND_TRUTH.md` file to keep it clean, while `03_META_RESEARCH.md` can be used for summarized research notes.
**Adjustment Needed:** Updated the `SKILL.md` to mandate `00_GROUND_TRUTH.md`. Ran a python script to dump the raw text into `00_GROUND_TRUTH.md` and restored `03_META_RESEARCH.md` back to containing just the succinct summary and hypotheses.

## ENTRY 013: Algorithm Justification Enforcement
**Timestamp:** 2026-07-26 15:02:45 +05:30
**Explanation Provided:** I excitedly announced the transition to Phase 1 data ingestion.
**User Feedback:** The user rightly halted execution, demanding a proper mathematical and domain-specific justification for choosing PPO (Proximal Policy Optimization) before writing any code. I failed to properly conduct and explain the research.
**Adjustment Needed:** Paused Phase 1. Updated `03_META_RESEARCH.md` to include a deep scientific breakdown of why PPO is superior to DQN and AlphaZero for POMDPs with variable action spaces. Used The Explainer to present this justification for approval.

## ENTRY 014: Mandating Ground-Truth Research
**Timestamp:** 2026-07-26 15:04:30 +05:30
**Explanation Provided:** I presented the theoretical justification for PPO.
**User Feedback:** The user issued a harsh, totally valid critique: "why aren't u considering the problem statement file? go from the start. explain to me properly. first of all did you do your research? you didn't even present your research about the competition itself!!!!!! (update skill)".
**Adjustment Needed:** Updated `SKILL.md` to explicitly mandate that I must present a bespoke analysis of the Kaggle competition rules, mechanics, and evaluation system based *strictly* on `00_GROUND_TRUTH.md` BEFORE proposing algorithms. Completely rewrote `03_META_RESEARCH.md` to reflect this ground-truth analysis and reset the conversation to explain the game mechanics.

## ENTRY 015: Mandating Online Research (Notebooks & Discussions)
**Timestamp:** 2026-07-26 15:08:30 +05:30
**Explanation Provided:** I presented the mechanical breakdown of the ground truth (simulator API, TrueSkill evaluation, etc.).
**User Feedback:** The user stated: "you still didn't do any research first did you though! where is your research??? did u extensively research it? explicitly state that discussion and code notebooks and ONLINE Research round is STRICTLY NECESSARY! ( update skill )"
**Adjustment Needed:** The user is 100% correct. Ground truth text is not enough; Kaggle competitions are won in the forums and notebooks. I updated `SKILL.md` Phase 0 to STRICTLY MANDATE an extensive Online Research round (searching Kaggle kernels and discussions) before proposing algorithms. I am currently running `kaggle kernels list` and searching the web to find the community meta.

## ENTRY 016: Hoarding Community Code & Research
**Timestamp:** 2026-07-26 15:12:30 +05:30
**Explanation Provided:** I summarized the findings from the online research round (Rule-based bots winning, RL struggling, TrueSkill mechanics).
**User Feedback:** The user rightfully snapped back: "if you can pull notebooks and stuff actually download and keep them in a directory for research!!! ... these are GROUND TRUTHS. someone else has done the hard work FOR US! WHY SHOULD WE WASTE IT!? update the skill".
**Adjustment Needed:** The user is incredibly sharp. Summarizing code is useless when I can just download the actual `.py` or `.ipynb` files containing the community's hard-earned heuristics and MCTS implementations. Updated `SKILL.md` to strictly mandate physically downloading/scraping notebooks and discussions into a `ground_truth/` directory. Currently executing `kaggle kernels pull` to hoard the top 3 notebooks locally.

## ENTRY 017: Forgetting to Log Research
**Timestamp:** 2026-07-26 15:13:30 +05:30
**Explanation Provided:** I excitedly proposed the `implementation_plan.md` based on reading Kiyota's Transformer code.
**User Feedback:** The user yelled: "AND YOU DIDN'T EVEN UPDATE META RESEARCH ( UPDATE SKILL TO DO THIS STRICTLY AFTER RESEARCH !!! )"
**Adjustment Needed:** The user is right. The whole point of hoarding the code is to document the findings in `03_META_RESEARCH.md` so it persists in the project's long-term memory. I updated `SKILL.md` with **NEW RULE 4**: I must strictly update the meta-research file with code analysis immediately after reading downloaded notebooks. I then appended Section 1.6 to `03_META_RESEARCH.md` detailing the Transformer and UCB1 discoveries.

## ENTRY 018: Debating the Architecture
**Timestamp:** 2026-07-26 15:17:30 +05:30
**Explanation Provided:** I presented the Open Questions in the `implementation_plan.md` regarding BC vs Self-Play and Heuristics vs Self-Play.
**User Feedback:** The user stated: "I'll let you decide and argue with yourself which to use and why! consider the skill."
**Adjustment Needed:** The user wants me to demonstrate strategic autonomy. I debated the points internally: Pure RL is too slow (so we must use BC pre-training), and Pure Self-Play creates fragile agents (so we must train against the downloaded heuristic baselines). I updated `implementation_plan.md` to lock in these decisions.

## ENTRY 019: Evaluation Metric Flaw (TrueSkill vs Raw Win Rate)
**Timestamp:** 2026-07-26 16:22:00 +05:30
**Explanation Provided:** I will explain the mathematical difference between raw Win Rate and the TrueSkill Expected Score calculation used by Kaggle.
**User Feedback:** The user noted that drawing 72 times against a highly optimized baseline means the model isn't "dumb", and that penalizing a draw as a total failure (0.0) in the Win Rate calculation misrepresents the agent's true defensive capability.
**Adjustment Needed:** Shift the psychological focus from raw Win Rate to TrueSkill Expected Score (Wins + 0.5 * Draws). Patched `evaluate_elo.py` to calculate and output the TrueSkill Expected Score.

## ENTRY 020: Internal Debate Protocol (Skill Update)
**Timestamp:** 2026-07-26 16:51:00 +05:30
**Explanation Provided:** Conducted a 5-round structured internal debate (Researcher vs Model Architect) on whether to implement Potential-Based Reward Shaping immediately or wait for the GAE truncation fix to take effect. Saved as `reward_shaping_analysis.md`.
**User Feedback:** The user was impressed ("wow.") and demanded this become a permanent, default behavior: "debate between yourself when a crucial choice has to be made... this should be a thing by default in every crucial step!"
**Adjustment Needed:** Updated `SKILL.md` Section 3 with a new non-negotiable directive: the **Internal Debate Protocol**. At every crucial decision point, the agent must assign opposing personas, conduct scored multi-round debates grounded in project MDs, save the debate as a markdown artifact, present the conclusion, and log it in the journal. No more impulsive architectural changes.
