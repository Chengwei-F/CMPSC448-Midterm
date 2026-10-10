"""Check the delivered data without installing packages or training a model."""
import collections
import csv
import json
import pathlib

csv.field_size_limit(2**31 - 1)
root = pathlib.Path(__file__).resolve().parent / "prepared"
audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
seen_groups = {}
seen_records = set()
seen_prompts = set()
total = 0
for split in ["train", "validation", "test"]:
    prompts = collections.defaultdict(list)
    counts = collections.Counter()
    with (root / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            assert row["split"] == split
            assert row["record_id"] not in seen_records
            seen_records.add(row["record_id"])
            group = row["group_id"]
            assert seen_groups.setdefault(group, split) == split
            prompts[row["prompt_id"]].append(row)
            counts[row["LLM_name"]] += 1
    assert not seen_prompts.intersection(prompts)
    seen_prompts.update(prompts)
    assert len(set(counts.values())) == 1
    assert set(counts) == set(audit["models"].values())
    for rows in prompts.values():
        assert len(rows) == 5
        assert len({r["model_id"] for r in rows}) == 5
        assert len({r["LLM_Input"] for r in rows}) == 1
        assert all(r["LLM_output"].strip() for r in rows)
    n = sum(counts.values())
    total += n
    assert n == audit["split_rows"][split]
    print(f"{split}: {len(prompts):,} prompts, {n:,} responses; {dict(counts)}")
assert total == audit["rows"]
print(f"PASS: {total:,} responses; {len(seen_prompts):,} prompts; {len(seen_groups):,} split groups")
print("Labels: " + str(audit["label_mapping"]))
print("Input-only deterministic accuracy must equal 20% on complete five-model groups.")
print("No classifier has been trained by this script.")
