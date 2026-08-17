import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import run

STRATEGIES = ["fixed", "structure", "semantic"]

summaries = {}
for s in STRATEGIES:
    print(f"\n########## running strategy: {s} ##########")
    summary, _ = run(strategy=s, top_k=15)
    summaries[s] = summary

METRICS = [
    ("correctness_overall",        "Answer correctness"),
    ("no_answer_decline_rate",     "No-answer decline"),
    ("retrieval_hit_overall",      "Retrieval hit"),
    ("retrieval_coverage_overall", "Retrieval coverage"),
    ("faithfulness_overall",       "Faithfulness"),
    ("citation_accuracy_overall",  "Citation accuracy"),
]

print("\n" + "=" * 66)
print("STRATEGY COMPARISON  (higher is better for every metric)")
header = f"{'metric':<20}" + "".join(f"{s:>12}" for s in STRATEGIES) + f"{'winner':>12}"
print(header)
print("-" * len(header))
for key, label in METRICS:
    vals = {s: (summaries[s][key] if summaries[s][key] is not None else 0.0)
            for s in STRATEGIES}
    winner = max(vals, key=vals.get)
    print(f"{label:<20}" + "".join(f"{vals[s]:>12.3f}" for s in STRATEGIES) + f"{winner:>12}")

print("\nCorrectness by question type:")
header = f"{'type':<12}" + "".join(f"{s:>12}" for s in STRATEGIES) + f"{'winner':>12}"
print(header)
print("-" * len(header))
for t in ["lookup", "multi_hop", "ambiguous"]:
    vals = {s: (summaries[s]["correctness_by_type"].get(t) or 0.0) for s in STRATEGIES}
    winner = max(vals, key=vals.get)
    print(f"{t:<12}" + "".join(f"{vals[s]:>12.3f}" for s in STRATEGIES) + f"{winner:>12}")
