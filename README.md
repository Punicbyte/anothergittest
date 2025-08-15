
# RRT & CSFT Eval Starter Kit (Minimal, Offline)

This kit gives you a **base test harness** to evaluate reward-model ideas (RRT or CSFT) before heavy training.
It focuses on **metrics + stress tests** and lets you plug in your own scorer later.

## What you get
- `eval.py` — runs metrics and stress tests given a dataset and a scoring function
- `scorers.py` — simple baseline scorers (length, ops-density, naive-final-answer)
- `metrics.py` — pairwise accuracy, FPR@TPR, score-length correlation (Pearson r), plus stress-test runners
- `data/example_data.jsonl` — tiny toy dataset with GOOD / SRDA / MIXED / WRONG examples
- `data/schema.md` — data format spec

## Quickstart
```bash
python eval.py --data data/example_data.jsonl --scorer length --split_seed 42
python eval.py --data data/example_data.jsonl --scorer ops_density --split_seed 42
python eval.py --data data/example_data.jsonl --scorer naive_final_answer --split_seed 42

# Stress tests
python eval.py --data data/example_data.jsonl --scorer ops_density --stress number_masked
python eval.py --data data/example_data.jsonl --scorer ops_density --stress answer_removed
```

## Plug in your own scorer
Open `scorers.py` and implement `score_with_model(question, answer, text)` to call your own model or judge.
Then run:
```bash
python eval.py --data data/your_data.jsonl --scorer model
```

## Metrics reported
- Pairwise accuracy (good > bad) on held-out test questions
- FPR@TPR (False Positive Rate at a fixed True Positive Rate) with threshold chosen on dev
- Score–length correlation (Pearson r), de-meaned within question to catch length bias
- Stress tests:
  - number-masked: replace digits with placeholders <n1>, <n2>, ...
  - answer-removed: drop trailing "Answer: ..." / "Therefore, ..." line from all candidates

## Data note
This kit expects per-question candidate sets with label in {"good","srda","mixed","wrong"}.
We auto-generate all pairs (good vs bad) within the same question for pairwise accuracy.
See `data/schema.md` for details.
