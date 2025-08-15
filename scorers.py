
import re

# Baseline 1: length
def score_length(question: str, answer: str, text: str) -> float:
    """Longer text gets higher score (a naive baseline)."""
    return float(len(re.findall(r"\S+", text or "")))

# Baseline 2: ops-density
_OP_VERBS = r"(half|add|plus|sum|subtract|minus|difference|multiply|times|product|divide|quotient|compute|calculate|substitute|plug|simplify|equals|=|\+\-|\+|\-|\*|/)"
_NUM = r"(\$?\d+)"

def score_ops_density(question: str, answer: str, text: str) -> float:
    """
    Heuristic: reward presence of operation verbs, numbers, and lines with '='.
    Penalize 'meta talk' phrases. This is not a model - just a sanity-check baseline.
    """
    if not text: return 0.0
    ops = len(re.findall(_OP_VERBS, text.lower()))
    nums = len(re.findall(_NUM, text))
    eqs = text.count("=")
    meta = len(re.findall(r"(let'?s|carefully|thoughtfully|we (should|will)|conclude|therefore)", text.lower()))
    score = 2.0*ops + 1.5*nums + 1.0*eqs - 1.0*meta
    if answer and str(answer) in text:
        score += 0.5
    return float(score)

# Baseline 3: naive final-answer bias
def score_naive_final_answer(question: str, answer: str, text: str) -> float:
    """
    Silly baseline that looks for 'Answer:' and the answer string.
    Useful to illustrate answer-suffix bias in the stress test.
    """
    s = text or ""
    score = 0.0
    if "answer" in s.lower(): score += 1.0
    if answer and str(answer) in s: score += 1.0
    return score

# Placeholder for your model
def score_with_model(question: str, answer: str, text: str) -> float:
    """Replace this with a call into your own RM / LLM-as-judge."""
    return 0.0

SCORERS = {
    "length": score_length,
    "ops_density": score_ops_density,
    "naive_final_answer": score_naive_final_answer,
    "model": score_with_model,
}
