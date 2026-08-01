# PPO Pipeline & Codebase Audit Report

This report consolidates the findings of the specialized subagent reviews across the PPO pipeline, Neural Network Architecture, Environment wrapping, Rule-Based test bots, and Hyperparameter tuning. We have uncovered several critical flaws that are actively sabotaging the learning process.

## 🚨 CRITICAL Severity

### 1. The "Eternal Early Game" Reset Loop
- **File:** `experiments/02_ppo_training/train_ppo.py`
- **Issue:** Inside the `global_update_count` loop, `obs_list = env.reset(opponent_name)` is called at the start of *every* 64-step epoch.
- **Why it's disastrous:** By hard-resetting all environments every 64 steps, the agent never plays a game past step 64. Pokémon TCG matches take hundreds of steps. The agent will never experience a real win or loss, and the Generalized Advantage Estimation (GAE) math will bootstrap off a truncated state that has actually been wiped.
- **Proposed Fix:** Move the `obs_list = env.reset(...)` call entirely *outside* the training loop. Allow `VectorEnv` to run continuously and rely on its internal auto-reset when an environment hits `done=True`.

### 2. Passing Lists of Tensors to `torch.tensor()`
- **File:** `experiments/02_ppo_training/train_ppo.py`
- **Issue:** The mini-batch creation loop uses `torch.tensor(mb_old_log_probs, device=device)`.
- **Why it's disastrous:** `mb_old_log_probs` is a list of scalar Tensors that already exist on the GPU. Passing a list of Tensors to `torch.tensor()` will raise a `ValueError` in modern PyTorch and crash the script instantly upon reaching the first optimization phase.
- **Proposed Fix:** Delete the `.append()` loop entirely. Since `chunk` is a list of integers, use PyTorch advanced indexing directly on the parent tensors: `old_lp = old_log_probs[torch.tensor(chunk)]`.

### 3. `AttributeError` from `.size` on Python Lists
- **File:** `experiments/02_ppo_training/train_ppo.py`
- **Issue:** Extracting legal action counts using `legal_count = buffer.dec_offsets[idx].size`.
- **Why it's disastrous:** `buffer.dec_offsets` is populated by lists returned by the environment wrappers. Python lists do not have a `.size` attribute. This will crash the script with an `AttributeError` during the very first PPO update.
- **Proposed Fix:** Replace `.size` with `len()`. Use `legal_count = len(buffer.dec_offsets[idx])`.

### 4. Critic Head Bounded by Tanh Activation
- **File:** `experiments/02_ppo_training/model.py`
- **Issue:** The `self.critic_head` initialization concludes with `nn.Tanh()`.
- **Why it's disastrous:** `Tanh()` artificially bounds the network's value predictions strictly to `(-1, 1)`. Outputting via `Tanh` causes severe gradient saturation (vanishing gradients) as predictions approach the limits, crippling the critic's ability to accurately calculate PPO advantages ($A = Q - V$).
- **Proposed Fix:** Remove `nn.Tanh()`. The final layer of a Value head must be a raw linear layer.

### 5. `self_play_agent` ignores `minCount`
- **File:** `experiments/03_rule_based/self_play_agent.py`
- **Issue:** Uses `action_idx = torch.argmax(probs).item()` and returns `[action_idx]`.
- **Why it's disastrous:** It completely ignores `obs.select.minCount`. If the Kaggle engine enters a state requiring multiple selections (e.g., discarding cards from hand or selecting multiple prizes), returning a list of length 1 violates the strict validation constraint, causing the bot to forfeit instantly.
- **Proposed Fix:** Implement a safety fallback for multiple-selection contexts: `if min_count > 1: return random.sample(list(range(len(options))), count)`.

---

## 🛑 HIGH Severity

### 6. Missing Scaling Factor in Actor Logits (Unscaled Dot-Product)
- **File:** `experiments/02_ppo_training/model.py`
- **Issue:** The actor network maps the state to the action space using a dot product: `torch.bmm(global_context, padded_actions.transpose(1, 2))` without dividing by `math.sqrt(self.d_model)`.
- **Why it's bad:** Without this scaling factor, the dot product's variance scales directly with the embedding dimension (128). At initialization, this causes extreme logit values, leading to early entropy collapse (the softmax becomes highly peaked immediately), which ruins exploration and causes actor gradients to vanish.
- **Proposed Fix:** Divide the output of the `bmm` by `math.sqrt(self.d_model)`.

### 7. Suicide by Invalid Action (Reward Shaping)
- **File:** `experiments/02_ppo_training/env_wrapper.py`
- **Issue:** The reward logic checks `if engine_reward == 1: r += 1.0; elif engine_reward == -1: r -= 1.0; else: r -= 0.1`.
- **Why it's bad:** Kaggle environments terminate games with a `None` or `0` reward (and status `"INVALID"`) when an agent submits an illegal action. Under the current logic, an invalid action bypasses the `-1.0` penalty and triggers the `else` block (`-0.1`). The PPO agent will quickly learn to intentionally commit illegal moves to gracefully crash the environment and escape the -1.0 penalty of losing.
- **Proposed Fix:** Read `info.get('status')` from the Kaggle environment. If the status is `ERROR` or `INVALID`, punish it with `r -= 1.0`.

### 8. Severe Data Starvation & Truncation (Hyperparameters)
- **File:** `experiments/02_ppo_training/train_ppo.py`
- **Issue:** `NUM_ENVS = 8` and `STEPS_PER_ENV = 64` yields a rollout buffer of only 512 steps. 
- **Why it's bad:** A Transformer-based PPO agent requires massive amounts of data per update. 512 steps is severely undersized. Furthermore, with `STEPS_PER_ENV = 64`, almost all rollouts will end prematurely before the match concludes. The agent will have to bootstrap heavily from `next_values`, which, early in training, is pure noise.
- **Proposed Fix:** Increase `NUM_ENVS` to 16, and `STEPS_PER_ENV` to 128 (yielding 2048 steps per update). 

### 9. Missing Gradient Clipping
- **File:** `experiments/02_ppo_training/train_ppo.py`
- **Issue:** `MAX_GRAD_NORM` is not defined, and gradient clipping is absent.
- **Why it's bad:** Transformers are prone to gradient explosions. Without clipping, a single bad batch will cause massive gradient spikes, leading to NaN weights.
- **Proposed Fix:** Apply `torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)` before `optimizer.step()`.

---

## ⚠️ MEDIUM/LOW Severity

### 10. Metric Erasure and Phantom 0% Win Rates
- **File:** `experiments/02_ppo_training/train_ppo.py`
- **Issue:** Episode metrics are calculated solely on the current batch and flushed every epoch.
- **Proposed Fix:** Use a sliding window (`collections.deque(maxlen=100)`) for global `wins` and `matches`.

### 11. Lack of Mini-batch Shuffling
- **File:** `experiments/02_ppo_training/train_ppo.py`
- **Issue:** Chunking the rollout buffer sequentially causes high gradient variance.
- **Proposed Fix:** Generate a random permutation of indices (`torch.randperm`) to shuffle the mini-batches.

### 12. Missing `sqrt(d_model)` Scaling for Token Embeddings
- **File:** `experiments/02_ppo_training/model.py`
- **Issue:** Input token embeddings are not scaled by $\sqrt{d_{\text{model}}}$ before adding positional encodings.
- **Proposed Fix:** Multiply token embeddings by `math.sqrt(self.d_model)` before adding the positional encodings.

### 13. Invalid Duplicate Indices in Bot Fallbacks
- **File:** `experiments/03_rule_based/*.py`
- **Issue:** The `except Exception` blocks in all bots fallback to `return [0] * count`.
- **Why it's bad:** The Kaggle engine prohibits duplicate elements in the action list. Returning `[0, 0]` guarantees an environment crash.
- **Proposed Fix:** Fallback to `list(range(min(max_count, len(options))))`.

---

## ✅ User Review Required
Please review the complete synthesized findings above. Once you approve, I will systematically execute these patches across the codebase.
