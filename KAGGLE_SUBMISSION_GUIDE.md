# Kaggle Submission Guide: Pokémon TCG AI Battle Challenge

This guide outlines the critical constraints, debugging strategies, and exact packaging requirements necessary to successfully submit an agent to the Pokémon TCG AI Battle Challenge on Kaggle without encountering `EpisodeState.ERROR`.

## 1. Environment Quirks & Constraints

The Kaggle validation environment differs from local development in several strict ways:

### Dependency Packaging
- **`cg` Module is NOT Global:** The `cg` API provided by the Kaggle `sample_submission` is **NOT** available globally in the Kaggle environment. 
- **Solution:** You must explicitly bundle the `cg/` directory inside your `submission.tar.gz` at the root level so that `import cg` resolves successfully.

### The Zero-Option Edge Case
- **Strict Length Checking:** The C++ game engine strictly enforces `minCount <= len(selected_actions) <= maxCount`.
- **The Trap:** During specific game states where no action is possible, Kaggle passes `maxCount == 0` and `len(option) == 0`. If your agent has a default fallback (e.g., picking `[0]` to avoid empty lists), the engine will evaluate `minCount(0) <= len([0])(1) <= maxCount(0)`, which returns `False` and crashes the environment.
- **Solution:** You must handle `maxCount == 0` or `len(option) == 0` by explicitly returning `[]`.

### The Step 0 Deck Submission
- **Initial State:** On Step 0, Kaggle passes `obs.select == None` expecting the agent to return the initial 60-card deck list (array of ints).
- **File Parsing:** The `deck.csv` must be bundled at the root of your tarball, and your agent should fall back to reading from `/kaggle_simulations/agent/deck.csv` if local paths fail.

## 2. Packaging the Submission

Your submission must be a `.tar.gz` file under 100 MiB.
The tarball must contain your agent entrypoint at the root level named `main.py`.

### Required Tarball Structure
```
submission.tar.gz
├── main.py                # Agent entry point
├── deck.csv               # 60-card initial deck
├── model.py               # (Optional) Model definitions
├── model_weights.pt       # (Optional) Saved checkpoint
└── cg/                    # The Kaggle sample_submission CG API folder
    ├── __init__.py
    ├── api.py
    ├── game.py
    └── ... (including .so / .dll files)
```

## 3. Debugging Kaggle Submission Crashes

When a submission gets a `SubmissionStatus.ERROR`, Kaggle provides replay and log artifacts that are essential for debugging:

1. **Get the Submission ID:**
   Run `kaggle competitions submissions -c pokemon-tcg-ai-battle` to view your submissions. Note the integer ID of the `ERROR` submission.
2. **Find the Episode ID:**
   Run `kaggle competitions episodes <SUBMISSION_ID>` to list the evaluation episodes. Note the Episode ID.
3. **Check the Replay JSON:**
   Run `kaggle competitions replay <EPISODE_ID>` to download the JSON replay. Check the `steps` array to see exactly which step marked `"status": "ERROR"` and read the `observation` object for that step.
4. **Check the Agent Logs (Stack Trace):**
   Run `kaggle competitions logs <EPISODE_ID> 0` (or `1` for the second agent) to download the agent's Python stack trace. This will definitively identify `ModuleNotFoundError` or other runtime exceptions.

## 4. The "Unbreakable Shell" Pattern

To survive edge cases, wrap your core agent logic in a `try...except` block and use the Kaggle sample submission's random choice as the absolute fallback.

```python
def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    
    # Handle Step 0 Deck Query
    if obs.select is None:
        return read_deck_csv()
        
    try:
        # VERY IMPORTANT: Handle Kaggle Zero-Option Edge Case
        if obs.select.maxCount == 0 or len(obs.select.option) == 0:
            return []
            
        # ... Your Complex Logic Here ...
        
        return calculated_action_list
        
    except Exception as e:
        # The Absolute Fallback
        return random.sample(list(range(len(obs.select.option))), obs.select.maxCount)
```
