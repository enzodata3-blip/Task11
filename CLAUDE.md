# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Context

This is the `model_b` workspace for **TASK_11251** — an A/B evaluation experiment comparing two Claude models on a machine learning optimization task. The experiment root is one directory up (`../`), where `model_a/` and `model_b/` are sibling workspaces.

- **This model (model_b):** Lion (`claude-churro-v10-p`)
- **Comparison model (model_a):** Buffalo (`claude-sonnet-4-5-20250929`)
- **Expert:** Enzo Rodriguez
- **Experiment manifest:** `../manifest.json`

## Task Objective

Translate and enhance a statistical/ML study originally written in R into Python. The core goals are:

1. **Replicate** the R-based model using Python (scikit-learn, statsmodels, pandas, numpy)
2. **Enhance** the model by reintroducing interaction terms identified via correlation matrix analysis
3. **Apply the human element**: inspect model performance, use correlation matrices to decide which variables/interactions to add back, and iterate
4. **Optimize** training/test data evaluation for robust generalization

The reference R library for broom-style tidy model output is [tidymodels/broom](https://github.com/tidymodels/broom). The Python equivalent is `statsmodels` (`.summary()`, `.tidy()` via `broom`-like wrappers) or `sklearn` pipelines.

## Session Logging (Hooks — Do Not Modify)

Hooks in `.claude/hooks/` automatically capture telemetry:

| Hook trigger | Script | What it does |
|---|---|---|
| `SessionStart` | `capture_session_event.py start` | Records git commit/branch at session start |
| `Stop` | `process_transcript.py incremental` | Appends new messages to the session log |
| `SessionEnd` | `process_transcript.py final` + `capture_session_event.py end` | Rebuilds full log, generates summary with token/tool/git metrics |

Logs are written to `../logs/model_b/session_<id>.jsonl` and `session_<id>_raw.jsonl`.

**Key utility (`claude_code_capture_utils.py`):** Detects model lane from CWD path, locates experiment root by looking for sibling `model_a`/`model_b` directories, reads `manifest.json` for task/assignment metadata, and routes log files to the correct model-specific subdirectory.

## Development Commands

Once ML code is added to this workspace, typical commands will be:

```bash
# Run a Python script
python <script_name>.py

# Install dependencies (if requirements.txt is created)
pip install -r requirements.txt

# Run tests (if pytest is configured)
pytest

# Run a single test
pytest tests/test_<name>.py::test_function_name -v
```

## Architecture Notes for ML Work

- Keep data loading, preprocessing, model training, and evaluation in clearly separated stages
- Store the correlation matrix analysis step explicitly — it is the decision point for which interaction terms to reintroduce
- When adding interaction terms, document *why* each term was chosen (correlation threshold, domain reasoning)
- Compare train vs. test metrics at each iteration to guard against overfitting from added interactions
- Use `statsmodels` when interpretability (p-values, confidence intervals, residuals) matters; use `sklearn` pipelines for cross-validation and hyperparameter tuning
