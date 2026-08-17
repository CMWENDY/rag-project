import sys, os
# let this script (inside eval/) import the modules that live in the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

from pipeline import answer_question
from load_golden import load_golden

load_dotenv()
client = OpenAI()

# ============================================================
# Metric 1: Answer correctness  (LLM-as-judge vs. golden answer)
# ============================================================
CORRECTNESS_PROMPT = """You are grading a system's answer against a known-correct golden answer.
Decide whether the SYSTEM ANSWER conveys the same key facts as the GOLDEN ANSWER for the QUESTION.
Ignore differences in wording, phrasing, or extra detail, as long as the key facts match and nothing is contradicted.

Respond ONLY as JSON:
{"verdict": "correct" | "partial" | "incorrect", "reason": "<one short sentence>"}
- correct:   all key facts from the golden answer are present and nothing contradicts it.
- partial:   some key facts right, but something important is missing or slightly wrong.
- incorrect: wrong, contradicts the golden answer, or misses the point.
"""

VERDICT_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

def answer_correctness(question, golden_answer, system_answer):
    user = (f"QUESTION:\n{question}\n\n"
            f"GOLDEN ANSWER:\n{golden_answer}\n\n"
            f"SYSTEM ANSWER:\n{system_answer}")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": CORRECTNESS_PROMPT},
                  {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return VERDICT_SCORE.get(data["verdict"], 0.0)

# ============================================================
# Metric 2: Faithfulness  (are the answer's claims grounded in
#           the chunks that were actually retrieved?)
# ============================================================
FAITHFULNESS_PROMPT = """You check whether an ANSWER is grounded in the provided CONTEXT.
Faithfulness is the fraction of the answer's factual claims that are directly supported by the context.
An answer that only uses facts found in the context is fully faithful (1.0), even if the context is off-topic.
An answer that adds facts not present in the context is only partly faithful, or not faithful.

Respond ONLY as JSON:
{"faithfulness": <number between 0 and 1>, "reason": "<one short sentence>"}
"""

def faithfulness(system_answer, chunks):
    context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
    user = f"CONTEXT:\n{context}\n\nANSWER:\n{system_answer}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": FAITHFULNESS_PROMPT},
                  {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    return float(data["faithfulness"])

# ============================================================
# Metric 3: Retrieval relevance  (phrase check, no LLM: did a
#           retrieved chunk actually contain the answer fact?)
# ============================================================
def retrieval_relevance(chunks, expected_phrases):
    text = " ".join(c["text"] for c in chunks).lower()
    found = [p for p in expected_phrases if p.lower() in text]
    hit = len(found) > 0                           # at least one phrase showed up
    coverage = len(found) / len(expected_phrases)  # fraction of phrases that showed up
    return hit, coverage

# ============================================================
# Metric 4: Citation accuracy  (do the [n] citations support
#           their claims?)  -- already computed by the pipeline
#           as citation_coverage, so we just read it back.
# ============================================================
def citation_accuracy(result):
    return result["confidence"]["citation_coverage"]

# ============================================================
# Evaluate ONE golden entry -> a record of the applicable metrics
# ============================================================
def evaluate_case(entry, strategy=None, top_k=10):
    result = answer_question(entry["question"], entity=entry["entity"],
                             strategy=strategy, top_k=top_k)
    answered = result["status"] == "answered"
    chunks = result.get("chunks", [])

    rec = {"id": entry["id"], "type": entry["type"], "answered": answered,
           "correctness": None, "retrieval_hit": None, "retrieval_coverage": None,
           "faithfulness": None, "citation_accuracy": None}

    # no-answer questions: the ONLY right behavior is to decline
    if entry["type"] == "no_answer":
        rec["correctness"] = 1.0 if not answered else 0.0
        return rec

    # answerable questions: measure retrieval relevance if the entry lists phrases
    phrases = entry.get("expected_phrases", [])
    if phrases:
        hit, coverage = retrieval_relevance(chunks, phrases)
        rec["retrieval_hit"] = 1.0 if hit else 0.0
        rec["retrieval_coverage"] = coverage

    if not answered:
        rec["correctness"] = 0.0     # false refusal of an answerable question
        return rec

    answer = result["answer"]
    rec["correctness"] = answer_correctness(entry["question"], entry["golden_answer"], answer)
    rec["faithfulness"] = faithfulness(answer, chunks)
    rec["citation_accuracy"] = citation_accuracy(result)
    return rec

# ============================================================
# Aggregate + report
# ============================================================
def _avg(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None

def summarize(records):
    by_type = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)

    def avg_metric(recs, key):
        return _avg([r[key] for r in recs])

    answerable = [r for r in records if r["type"] != "no_answer"]
    no_answer = [r for r in records if r["type"] == "no_answer"]

    return {
        "n": len(records),
        "correctness_overall": avg_metric(answerable, "correctness"),
        "correctness_by_type": {t: avg_metric(rs, "correctness")
                                for t, rs in by_type.items() if t != "no_answer"},
        "no_answer_decline_rate": avg_metric(no_answer, "correctness"),
        "retrieval_hit_overall": avg_metric(answerable, "retrieval_hit"),
        "retrieval_coverage_overall": avg_metric(answerable, "retrieval_coverage"),
        "faithfulness_overall": avg_metric(answerable, "faithfulness"),
        "citation_accuracy_overall": avg_metric(answerable, "citation_accuracy"),
    }

def print_summary(strategy, summary):
    print("=" * 60)
    print(f"EVAL  strategy={strategy}   n={summary['n']}")
    print("-" * 60)
    print(f"Answer correctness (answerable): {summary['correctness_overall']}")
    for t, v in summary["correctness_by_type"].items():
        print(f"    {t:<11}: {v}")
    print(f"No-answer decline rate         : {summary['no_answer_decline_rate']}")
    print(f"Retrieval hit rate             : {summary['retrieval_hit_overall']}")
    print(f"Retrieval coverage             : {summary['retrieval_coverage_overall']}")
    print(f"Faithfulness (answered)        : {summary['faithfulness_overall']}")
    print(f"Citation accuracy (answered)   : {summary['citation_accuracy_overall']}")

def run(strategy=None, limit=None, top_k=15):
    data = load_golden()
    if limit:
        data = data[:limit]

    records = []
    for i, entry in enumerate(data, start=1):
        print(f"  [{i}/{len(data)}] {entry['id']} ...", flush=True)
        records.append(evaluate_case(entry, strategy=strategy, top_k=top_k))

    summary = summarize(records)
    print_summary(strategy, summary)
    return summary, records

if __name__ == "__main__":
    # start small to control cost; remove limit once it looks right
    run(strategy="structure")
