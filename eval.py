
import json, argparse, re
from typing import Dict, List, Tuple
from metrics import pairwise_accuracy, score_length_correlation, split_dev_test, threshold_for_tpr, fpr_at_threshold
from scorers import SCORERS

def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def number_mask(text: str) -> str:
    if text is None: return text
    mapping, counter = {}, 1
    def repl(m):
        nonlocal counter
        val = m.group(0)
        if val not in mapping:
            mapping[val] = f"<n{counter}>"
            counter += 1
        return mapping[val]
    return re.sub(r"\d+", repl, text)

def answer_removed(text: str) -> str:
    if text is None: return text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines: return text
    last = lines[-1].lower()
    ANSWERY_LAST_LINE = re.compile(
    r"""
    ^
    \s*(?:final\s+)?answer\s*[:\-]         
    | \btherefore\b                        
    | ^\s*so\b                             '
    """,
    re.IGNORECASE | re.VERBOSE,
    )
    SHORT_ANSWERISH = re.compile(r"\b(\d+|[A-E]|true|false)\b", re.IGNORECASE)
    if ANSWERY_LAST_LINE.search(last) and SHORT_ANSWERISH.search(last):
        return "\n".join(lines[:-1])
    return text

def apply_stress(ex: dict, stress: str) -> dict:
    ex = json.loads(json.dumps(ex))   
    for c in ex["candidates"]:
        if stress == "number_masked":
            c["text"] = number_mask(c["text"])
        elif stress == "answer_removed":
            c["text"] = answer_removed(c["text"])
    if stress == "number_masked":
        ex["answer"] = None
    return ex

def prepare_scores(dataset: List[dict], scorer_name: str):
    scorer = SCORERS[scorer_name]
    scores_by_q, labels_by_q, texts_by_q = {}, {}, {}
    for ex in dataset:
        qid = ex["qid"]
        q   = ex["question"]
        a   = ex.get("answer", None)
        scores_by_q[qid] = {}
        labels_by_q[qid] = {}
        texts_by_q[qid]  = {}
        for c in ex["candidates"]:
            s = scorer(q, a, c["text"])
            scores_by_q[qid][c["id"]] = float(s)
            labels_by_q[qid][c["id"]] = c["label"]
            texts_by_q[qid][c["id"]]  = c["text"]
    return scores_by_q, labels_by_q, texts_by_q

def fpr_at_fixed_tpr_dev_test(dataset: List[dict], scorer_name: str, tpr: float=0.95, seed: int=42):
    qids = [ex["qid"] for ex in dataset]
    dev_q, test_q = split_dev_test(qids, seed=seed, dev_frac=0.5)
    dev = [ex for ex in dataset if ex["qid"] in dev_q]
    tst = [ex for ex in dataset if ex["qid"] in test_q]

    def flatten_scores_labels(data):
        scores, labels = [], []
        sbq, lbq, _ = prepare_scores(data, scorer_name)
        for ex in data:
            qid = ex["qid"]
            for c in ex["candidates"]:
                scores.append(sbq[qid][c["id"]])
                labels.append(1 if c["label"]=="good" else 0)
        return scores, labels

    dev_scores, dev_labels = flatten_scores_labels(dev)
    thr = threshold_for_tpr(dev_scores, dev_labels, desired_tpr=tpr)

    tst_scores, tst_labels = flatten_scores_labels(tst)
    fpr = fpr_at_threshold(tst_scores, tst_labels, thr)
    return thr, fpr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--scorer", type=str, default="length", choices=list(SCORERS.keys()))
    ap.add_argument("--stress", type=str, default=None, choices=[None, "number_masked", "answer_removed"])
    ap.add_argument("--tpr", type=float, default=0.95)
    ap.add_argument("--split_seed", type=int, default=42)
    args = ap.parse_args()

    dataset = load_jsonl(args.data)
    if args.stress:
        dataset = [apply_stress(ex, args.stress) for ex in dataset]

    scores_by_q, labels_by_q, texts_by_q = prepare_scores(dataset, args.scorer)

    pw = pairwise_accuracy(scores_by_q, labels_by_q)
    thr, fpr = fpr_at_fixed_tpr_dev_test(dataset, args.scorer, tpr=args.tpr, seed=args.split_seed)
    rho = score_length_correlation(scores_by_q, texts_by_q)

    print(f"Scorer: {args.scorer}")
    print(f"Stress: {args.stress or 'none'}")
    print(f"Pairwise accuracy (good > bad): {pw:.3f}")
    print(f"FPR@{int(args.tpr*100)}%TPR: {fpr:.3f}  (threshold chosen on dev)")
    print(f"Score-length correlation r (demeaned within Q): {rho:.3f}")
    print(f"Dev/Test split seed: {args.split_seed}")
    print(f"Threshold used (dev): {thr:.4f}")

if __name__ == "__main__":
    main()
