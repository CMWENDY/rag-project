import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import hybrid_search
from scorer import retrieval_confidence
from load_golden import load_golden

TOP_K, STRATEGY = 15, "structure"

rows = []
for i, e in enumerate(load_golden(), start=1):
    chunks = hybrid_search(e["question"], entity=e["entity"], strategy=STRATEGY, top_k=TOP_K)
    rc = float(retrieval_confidence(e["question"], chunks))
    rows.append((e["type"], e["id"], rc))
    print(f"  [{i}/63] {e['id']}  rc={rc:.3f}", flush=True)

noans      = sorted([r for r in rows if r[0] == "no_answer"], key=lambda x: x[2])
answerable = sorted([r for r in rows if r[0] != "no_answer"], key=lambda x: x[2])

def stats(name, g):
    v = [r[2] for r in g]
    print(f"{name} (n={len(v)}): min={min(v):.3f}  median={v[len(v)//2]:.3f}  max={max(v):.3f}")

print("\n" + "=" * 55)
print("NO-ANSWER questions (want these LOW so the gate declines them):")
for t, qid, rc in noans:
    print(f"  {rc:.3f}  {qid}")
print()
stats("NO-ANSWER ", noans)
stats("ANSWERABLE", answerable)

print("\nThreshold sweep  (a question is declined when rc < threshold):")
print("  thr  | no-answer declined (good) | answerable declined (bad)")
for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    good = sum(1 for _, _, rc in noans if rc < thr)
    bad  = sum(1 for _, _, rc in answerable if rc < thr)
    print(f"  {thr:.2f} |        {good}/{len(noans)}        |        {bad}/{len(answerable)}")
