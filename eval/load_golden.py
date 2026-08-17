import json
from collections import Counter

VALID_TYPES = {"lookup", "multi_hop", "no_answer", "ambiguous"}

def load_golden(path="eval/golden_dataset.json"):
    with open(path) as f:
        data = json.load(f)

    ids = [d["id"] for d in data]
    assert len(ids) == len(set(ids)), "duplicate ids found"

    for d in data:
        assert d["type"] in VALID_TYPES, f"{d['id']}: bad type {d['type']}"
        if d["type"] == "no_answer":
            assert d["expected_sources"] == [], \
                f"{d['id']}: no_answer must have empty expected_sources"
        else:
            assert d["expected_sources"], \
                f"{d['id']}: needs at least one expected_source"

    return data

if __name__ == "__main__":
    data = load_golden()
    print("total questions:", len(data))
    print("by type:", dict(Counter(d["type"] for d in data)))
    print("by entity:", dict(Counter(str(d["entity"]) for d in data)))
