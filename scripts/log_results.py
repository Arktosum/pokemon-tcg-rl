with open('01_JOURNEY_LOG.md', 'a', encoding='utf-8') as f:
    f.write(r"""
## Phase 18: Curriculum Learning & Domain Randomization Evaluation

**Execution Details:**
- Implemented Curriculum Rewards (Bench-filling and Energy Attachment delta bonuses).
- Enforced 30% Domain Randomization split against Random Agent in the PPO training loop.
- Scaled training to 3000 episodes (~12 minutes execution time).
- Mean Policy Entropy remained healthy (ended at ~0.81), confirming Entropy Collapse has been prevented.
- Value loss and Actor loss stabilized.

**Evaluation Results (Strict 500-Game Set):**
- Win Rate vs Random: 383/500 (76.6%)
- Win Rate vs Repaired Greedy: 262/500 (52.4%)

**Diagnostic:**
- **CRITERIA FAILED.** The >95% Random and >80% Greedy thresholds were missed.
- We have completely resolved numerical instability (NaNs), entropy collapse, and credit assignment bottlenecks, and expanded capacity via ResNet-256. 
- However, 76.6% win rate against Random implies the policy is still struggling to chain deep sequential actions together reliably, or it has converged to a local optimum where it prioritizes safe, low-risk plays over definitive winning strategies. 
""")

with open('02_EXPERIMENT_TRACKER.md', 'a', encoding='utf-8') as f:
    f.write(r"""
| Phase 18 | ResNet-256 + PPO + Curriculum (3000 ep) | 76.6% (383/500) | 52.4% (262/500) | FAILED (Under 95/80 target) |
""")
