
import numpy as np
import re
from collections import defaultdict
from typing import List, Dict, Tuple

def tokenize_len(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x)==0 or np.std(y)==0:
        return 0.0
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float((x * y).mean())

def pairwise_accuracy(scores_by_q: Dict[str, Dict[str, float]], labels_by_q: Dict[str, Dict[str, str]]) -> float:
    """For each question, compare all (good vs non-good) pairs; compute fraction where score_good > score_bad."""
    wins, total = 0, 0
    for qid, cand_scores in scores_by_q.items():
        labels = labels_by_q[qid]
        goods = [cid for cid,l in labels.items() if l=="good"]
        bads  = [cid for cid,l in labels.items() if l!="good"]
        for g in goods:
            for b in bads:
                total += 1
                if cand_scores.get(g, -1e9) > cand_scores.get(b, -1e9):
                    wins += 1
    return wins / total if total>0 else 0.0

def split_dev_test(qids: List[str], seed: int = 42, dev_frac: float = 0.5) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    qids = list(qids)
    rng.shuffle(qids)
    k = max(1, int(len(qids)*dev_frac))
    return qids[:k], qids[k:]

def threshold_for_tpr(scores: List[float], labels: List[int], desired_tpr: float=0.95):
    """Pick smallest threshold achieving TPR >= desired_tpr on dev. labels: 1=good, 0=bad."""
    uniq = sorted(set(scores), reverse=True)
    best_thr = None
    for thr in uniq + [min(uniq)-1e-6]:
        tp = sum(1 for s,l in zip(scores,labels) if l==1 and s>=thr)
        fn = sum(1 for s,l in zip(scores,labels) if l==1 and s<thr)
        tpr = tp / (tp+fn) if (tp+fn)>0 else 0.0
        if tpr >= desired_tpr:
            best_thr = thr
    if best_thr is None:
        best_thr = max(uniq)+1.0  # nothing meets TPR; degenerate
    return best_thr

def fpr_at_threshold(scores: List[float], labels: List[int], thr: float) -> float:
    fp = sum(1 for s,l in zip(scores,labels) if l==0 and s>=thr)
    tn = sum(1 for s,l in zip(scores,labels) if l==0 and s<thr)
    return fp / (fp+tn) if (fp+tn)>0 else 0.0

def score_length_correlation(scores_by_q: Dict[str, Dict[str, float]], texts_by_q: Dict[str, Dict[str, str]]) -> float:
    """Compute Pearson r between score and length, after de-meaning within each question."""
    diffs_scores, diffs_lens = [], []
    for qid, sdict in scores_by_q.items():
        lens = {cid: tokenize_len(texts_by_q[qid][cid]) for cid in sdict.keys()}
        svals = np.array(list(sdict.values()), dtype=float)
        lvals = np.array(list(lens.values()), dtype=float)
        if len(svals) < 2:
            continue
        diffs_scores.extend(list(svals - svals.mean()))
        diffs_lens.extend(list(lvals - lvals.mean()))
    if not diffs_scores:
        return 0.0
    return pearson_r(np.array(diffs_lens), np.array(diffs_scores))

def build_pairs(dataset: List[dict]):
    labels_by_q, texts_by_q = {}, {}
    for ex in dataset:
        qid = ex["qid"]
        labels_by_q[qid] = {c["id"]: c["label"] for c in ex["candidates"]}
        texts_by_q[qid]  = {c["id"]: c["text"]  for c in ex["candidates"]}
    return labels_by_q, texts_by_q

def flatten_items(dataset: List[dict]):
    ids, labels = [], []
    for ex in dataset:
        for c in ex["candidates"]:
            ids.append((ex["qid"], c["id"]))
            labels.append(1 if c["label"]=="good" else 0)
    return ids, labels
