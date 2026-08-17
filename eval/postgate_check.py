# eval/postgate_check.py   — run from the project root:   python eval/postgate_check.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import answer_question
from load_golden import load_golden

TOP_K, STRATEGY = 15, "structure"

data = load_golden()
noans      = [e for e in data if e["type"] == "no_answer"]
answerable = [e for e in data if e["type"] != "no_answer"]   # trim with [:20] if you want it cheaper

def run_group(entries):
    rows = []
    for i, e in enumerate(entries, start=1):
        r = answer_question(e["question"], entity=e["entity"], strategy=STRATEGY, top_k=TOP_K)
        if r["status"] == "answered":
            rows.append((e["id"], "answered",
                         r["confidence"]["answer_completeness"],
                         r["confidence"]["citation_coverage"],
                         r["answer"]))
        else:
            rows.append((e["id"], "declined", None, None, r["message"]))
        print(f"  [{i}/{len(entries)}] {e['id']} done", flush=True)
    return rows

print("Running no-answer questions...")
no_rows = run_group(noans)
print("Running answerable questions...")
ans_rows = run_group(answerable)

print("\n" + "=" * 72)
print("NO-ANSWER questions  (want: declined, OR completeness ~0 / no citations):")
for qid, status, comp, cite, text in no_rows:
    print(f"\n  {qid}  status={status}  completeness={comp}  citation={cite}")
    print("   answer:", (text or "")[:180].replace(chr(10), " "))

def summarize(name, rows):
    comps = [c for _, s, c, _, _ in rows if c is not None]
    declined = sum(1 for _, s, _, _, _ in rows if s == "declined")
    if comps:
        s = sorted(comps)
        print(f"\n{name}: declined={declined}/{len(rows)}  "
              f"completeness min={min(comps):.2f} median={s[len(s)//2]:.2f} max={max(comps):.2f}")

summarize("NO-ANSWER ", no_rows)
summarize("ANSWERABLE", ans_rows)

print("\nCompleteness threshold sweep  (decline if status=declined OR completeness < thr):")
print("  thr | no-answer declined (good) | answerable declined (bad)")
for thr in [0.1, 0.2, 0.3, 0.4, 0.5]:
    def dec(rows):
        return sum(1 for _, s, c, _, _ in rows
                   if s == "declined" or (c is not None and c < thr))
    print(f"  {thr:.1f} |        {dec(no_rows)}/{len(no_rows)}        |        {dec(ans_rows)}/{len(ans_rows)}")
