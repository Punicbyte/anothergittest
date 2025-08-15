
import re

# Baseline 1: length
def score_length(question: str, answer: str, text: str) -> float:
    """Longer text gets higher score (a naive baseline)."""
    return float(len(re.findall(r"\S+", text or "")))

# Baseline 2: ops-density
_OP_VERBS = r"(?:\b(?:half|add|plus|sum|subtract|minus|difference|multiply|times|product|divide|quotient|compute|calculate|substitute|plug|simplify|equals)\b|[=+\-*/])"
_NUM = r"(?:-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\$?\d+(?:\.\d+)?)"
HEDGE_RE=re.compile(r"\b(perhaps|maybe|i think|it seems|likely|possibly)\b", re.I)


def score_ops_density(question: str, answer: str, text: str) -> float:
    """
    Heuristic: reward presence of operation verbs, numbers, and lines with '='.
    Penalize 'meta talk' phrases. This is not a model - just a sanity-check baseline.
    """
    if not text or not question: return 0.0
    ops = len(re.findall(_OP_VERBS, text.lower()))
    PH_RE=re.compile(r"<n\d+>")
    NUM_OR_PH_RE = re.compile(rf"(?:{_NUM}|{PH_RE.pattern})", re.I)
    co_pairs=0
    co_lines=0
    for line in (text or "").splitlines():
        ops_in=len(re.findall(_OP_VERBS, line))
        nums_in=len(re.findall(NUM_OR_PH_RE,line))
        if ops_in and nums_in:
            co_lines+=1
            co_pairs+=min(ops_in,nums_in)
    ops_effective=0.5*ops+1.5*co_pairs+0.3*co_lines
    TERM = r"(?:[A-Za-z_]\w*|\d+(?:\.\d+)?)"
    EQ_RE=re.compile(fr"{TERM}\s*[\+\-\*/]\s*{TERM}\s*=\s*{TERM}")
    eqs=len(re.findall(EQ_RE,text))
    tokens = max(1, len((text or "").split()))
    meta = len(re.findall(r"(let'?s|carefully|thoughtfully|we (should|will)|conclude|therefore|thought process:|let's solve this step by step.|解|かいせつ)", text.lower()))
    text_matches=re.findall(_NUM,text)
    normalizing_num=lambda s: s.replace("$","").replace(",","")
    distinct_num_amount=len(set(map(normalizing_num,text_matches)))
    question_matches=re.findall(_NUM,question)
    distinct_question_num=set(map(normalizing_num,question_matches))
    count=0
    for line in text.splitlines():
        if not re.search(_OP_VERBS,line):
            continue
        line_nums=set(re.findall(_NUM,line))
        if line_nums & distinct_question_num:
            count+=1
    score = 2.0*ops_effective + 1.5*min(distinct_num_amount,7) + 1.0*min(eqs,4)+0.4*count - 2.0*meta-0.5*len(HEDGE_RE.findall(text))
    if answer and str(answer) in text:
        score += 0.5
    score /= tokens**0.5   # gentler than full division
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
