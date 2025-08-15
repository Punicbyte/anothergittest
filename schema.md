
# Data schema (JSONL)

Each line is one question with its candidate reasonings.

```json
{
  "qid": "string-question-id",
  "question": "text of the question",
  "answer": "optional short reference answer (string or null)",
  "candidates": [
    {"id": "cand1", "text": "raw reasoning or sketch text", "label": "good"},
    {"id": "cand2", "text": "...", "label": "srda"},
    {"id": "cand3", "text": "...", "label": "mixed"},
    {"id": "cand4", "text": "...", "label": "wrong"}
  ]
}
```

Labels
- good — operational, chained, correct (or clearly on track if no answer)
- srda — style-without-substance (semantic reasoning dilution / fluff)
- mixed — some ops but broken chain / irrelevant / sparse operations
- wrong — operational and chained but reaches incorrect result

Notes
- You can have multiple goods/bads per question.
- The kit forms all pairs (good vs non-good) within the same question for pairwise accuracy.
- For FPR@TPR, we treat all items individually: goods are positive class, others negative.
