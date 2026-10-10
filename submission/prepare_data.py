"""Prepare a paired five-family attribution dataset from LLMRouterBench.

No model calls. Original prompt and raw_output strings are preserved.
Use --archive for full reproducibility, or --selected for an extracted cache.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import pathlib
import random
import re
import tarfile
import unicodedata

MODELS = {
    "gpt-5-chat": "GPT",
    "claude-sonnet-4": "Claude",
    "gemini-2.5-flash": "Gemini",
    "qwen3-235b-a22b-2507": "Qwen",
    "deepseek-v3-0324": "DeepSeek",
}
SOURCES = {
    "aime/hybrid": ("aime", "math"),
    "livemathbench/test": ("livemathbench", "math"),
    "livecodebench/test": ("livecodebench", "coding"),
    "gpqa/test": ("gpqa", "knowledge_reasoning"),
    "hle/test": ("hle", "knowledge_reasoning"),
    "mmlupro/test_3000": ("mmlupro", "knowledge_reasoning"),
    "simpleqa/test": ("simpleqa", "factual_qa"),
    "arenahard": ("arenahard", "instruction_following"),
}
SEED = 20261005
REVISION = "0e5af1b84bf73437a01a1849c0f1d2468baa93fc"
URL = f"https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/{REVISION}/bench-release.tar.gz"
ARCHIVE_SHA256 = "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1"


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    temporary = pathlib.Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(temporary, path)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        self.parent[max(x, y)] = min(x, y)


def chosen_file(name):
    parts = pathlib.PurePosixPath(name).parts
    if parts and parts[0] == "bench-release":
        parts = parts[1:]
    if len(parts) < 3 or parts[-2] not in MODELS or not name.endswith(".json"):
        return None
    source = "/".join(parts[:-2])
    if source in SOURCES:
        return source, parts[-2], "/".join(parts)
    return None


def load_sources(args):
    selected = {}
    if args.archive:
        with tarfile.open(args.archive, "r|gz") as archive:
            for member in archive:
                key = chosen_file(member.name) if member.isfile() else None
                if key:
                    with archive.extractfile(member) as f:
                        raw = f.read()
                    source, model, relative = key
                    if (source, model) in selected:
                        raise ValueError(f"Multiple runs for {source}/{model}: choose explicitly")
                    selected[source, model] = (relative, json.loads(raw), hashlib.sha256(raw).hexdigest())
    else:
        for path in sorted(pathlib.Path(args.selected).rglob("*.json")):
            relative = path.relative_to(args.selected).as_posix()
            key = chosen_file(relative)
            if key:
                source, model, relative = key
                if (source, model) in selected:
                    raise ValueError(f"Multiple runs for {source}/{model}")
                raw = path.read_bytes()
                selected[source, model] = (relative, json.loads(raw), hashlib.sha256(raw).hexdigest())
    assert len(selected) == len(SOURCES) * len(MODELS), len(selected)
    return selected


def failure(text):
    return text.strip().casefold() == "failed to generate" or bool(re.match(r"\s*(?:Generation failed:|Error code:\s*\d{3}|Traceback \(most recent call last\)|APIError:|RateLimitError:)", text))


def flags(text):
    words = text.split()
    out = []
    if len(words) < 20:
        out.append("short_lt20_whitespace_words")
    if len(words) > 4096:
        out.append("long_gt4096_whitespace_words")
    if re.search(r"\b(?:openai|chatgpt|gpt-[345]|anthropic|claude|gemini|deepseek|qwen|alibaba)\b", text, re.I):
        out.append("model_name_mention_review")
    if re.search(r"\b(?:I (?:cannot|can't|am unable to)|I (?:do not|don't) have access|I'm sorry)\b", text, re.I):
        out.append("possible_refusal_review")
    if len(words) >= 100:
        trigrams = collections.Counter(zip(words, words[1:], words[2:]))
        if trigrams.most_common(1)[0][1] / (len(words) - 2) >= 0.15:
            out.append("repetitive_trigram_review")
    if "<think>" in text or "</think>" in text:
        out.append("think_tag_review")
    return out


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive")
    source.add_argument("--selected")
    parser.add_argument("--archive-sha256", default=ARCHIVE_SHA256, help="Known archive hash when using selected cache")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    archive_sha = file_hash(args.archive) if args.archive else args.archive_sha256
    if args.archive and archive_sha != ARCHIVE_SHA256:
        raise ValueError("Archive checksum differs from the frozen release; verify the download")
    output = pathlib.Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    files = load_sources(args)
    arena_map_path = pathlib.Path(__file__).resolve().parent / "arena_task_domains.json"
    arena_map = json.loads(arena_map_path.read_text(encoding="utf-8"))
    pairs, excluded, provenance = [], [], []
    for source, (benchmark, domain) in SOURCES.items():
        by_model = {}
        for model in MODELS:
            relative, data, sha = files[source, model]
            assert data["model_name"] == model
            recs = data["records"]
            lookup = {str(r["index"]): r for r in recs}
            assert len(lookup) == len(recs), (source, model, "repeated index")
            by_model[model] = lookup
            provenance.append({"source_file": relative, "sha256": sha, "records": len(recs), "model_id": model})
        indexes = set().union(*(set(r) for r in by_model.values()))
        for index in sorted(indexes, key=lambda x: (len(x), x)):
            reasons = []
            if any(index not in records for records in by_model.values()):
                reasons.append("missing_model")
            else:
                rows = [by_model[m][index] for m in MODELS]
                if not all(isinstance(r.get("prompt"), str) and r["prompt"].strip() for r in rows):
                    reasons.append("empty_or_nontext_prompt")
                elif len({r["prompt"] for r in rows}) != 1:
                    reasons.append("prompt_not_byte_identical")
                if not all(isinstance(r.get("raw_output"), str) and r["raw_output"].strip() for r in rows):
                    reasons.append("empty_or_nontext_output")
                for model, r in zip(MODELS, rows):
                    if isinstance(r.get("raw_output"), str) and failure(r["raw_output"]):
                        reasons.append("api_failure:" + model)
            if reasons:
                excluded.append({"source": source, "source_index": index, "reasons": ";".join(reasons)})
                continue
            first = rows[0]
            actual_domain = arena_map["prompt_domains"][digest(first["prompt"])] if benchmark == "arenahard" else domain
            origin = first.get("origin_query", first["prompt"])
            assert isinstance(origin, str) and origin.strip()
            assert len({r["origin_query"] for r in rows}) == 1, (source, index, "origin mismatch")
            pairs.append({"source": source, "benchmark": benchmark, "domain": actual_domain,
                          "index": index, "prompt": first["prompt"], "origin": origin,
                          "records": dict(zip(MODELS, rows))})
    pre_dedup = len(pairs)
    # Remove normalized identical prompts OR identical underlying questions globally.
    exact = UnionFind(len(pairs))
    seen = {}
    for i, p in enumerate(pairs):
        for kind, text in [("prompt", p["prompt"]), ("origin", p["origin"])]:
            key = kind, norm(text)
            if key in seen:
                exact.union(i, seen[key])
            else:
                seen[key] = i
    representatives = {}
    duplicates = []
    for i, p in enumerate(pairs):
        root = exact.find(i)
        if root in representatives:
            kept = representatives[root]
            duplicates.append({"removed_source": p["source"], "removed_index": p["index"],
                               "retained_source": kept["source"], "retained_index": kept["index"]})
        else:
            representatives[root] = p
    pairs = list(representatives.values())
    n = len(pairs)
    grouped = UnionFind(n)
    links = []
    # Group very similar underlying questions, retaining their rows in one split.
    # This preprocessing never sees labels/predictions and is not a predictive feature.
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [norm(p["origin"]) for p in pairs]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 max_features=120000, dtype=np.float32)
    matrix = vectorizer.fit_transform(texts)
    for start in range(0, n, 128):
        similarity = (matrix[start:start+128] @ matrix.T).tocoo()
        for local, j, value in zip(similarity.row, similarity.col, similarity.data):
            i = start + int(local)
            j = int(j)
            if j > i and value >= 0.95 and min(len(texts[i]), len(texts[j])) >= 80:
                grouped.union(i, j)
                links.append({"left_prompt_id": digest(pairs[i]["prompt"]),
                              "right_prompt_id": digest(pairs[j]["prompt"]),
                              "reason": "origin_char_tfidf_cosine_ge_0.95", "similarity": round(float(value), 6)})
    # Identical substantial output passages also stay in one split. Common short
    # factual answers intentionally do not create giant connected components.
    seen_long = {}
    for i, p in enumerate(pairs):
        for r in p["records"].values():
            out = norm(r["raw_output"])
            if len(out.split()) >= 50:
                h = digest(out)
                if h in seen_long and seen_long[h] != i:
                    j = seen_long[h]
                    grouped.union(i, j)
                    links.append({"left_prompt_id": digest(pairs[i]["prompt"]),
                                  "right_prompt_id": digest(pairs[j]["prompt"]),
                                  "reason": "identical_output_ge50_words", "similarity": 1.0})
                else:
                    seen_long[h] = i
    components = collections.defaultdict(list)
    for i in range(n):
        components[grouped.find(i)].append(i)
    # Stratify connected groups by benchmark composition, never split a group.
    strata = collections.defaultdict(list)
    for indexes in components.values():
        key = "+".join(sorted({pairs[i]["benchmark"] for i in indexes}))
        strata[key].append(indexes)
    rng = random.Random(SEED)
    assignments = {}
    for key in sorted(strata):
        groups = sorted(strata[key], key=lambda g: digest(pairs[g[0]]["prompt"]))
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)  # stable: random order within equal sizes
        total = sum(map(len, groups))
        targets = {"train": total * .7, "validation": total * .15, "test": total * .15}
        counts = dict.fromkeys(targets, 0)
        for indexes in groups:
            split = max(targets, key=lambda s: targets[s] - counts[s])
            counts[split] += len(indexes)
            group_id = digest("|".join(sorted(digest(pairs[i]["prompt"]) for i in indexes)))
            for i in indexes:
                assignments[i] = split, group_id
    data = collections.defaultdict(list)
    manifest, reviewed = [], []
    counter = collections.Counter()
    flag_counter = collections.Counter()
    for i, p in enumerate(pairs):
        split, group_id = assignments[i]
        pid = digest(p["prompt"])
        manifest.append({"prompt_id": pid, "group_id": group_id, "split": split,
                         "benchmark": p["benchmark"], "task_domain": p["domain"],
                         "source_index": p["index"], "origin_sha256_normalized": digest(norm(p["origin"]))})
        for label, (model, family) in enumerate(MODELS.items()):
            raw = p["records"][model]["raw_output"]
            qflags = flags(raw)
            relative = files[p["source"], model][0]
            rid = digest(relative + ":" + p["index"])
            row = {"record_id": rid, "prompt_id": pid, "group_id": group_id,
                   "LLM_name": family, "model_id": model, "label": label,
                   "LLM_Input": p["prompt"], "LLM_output": raw,
                   "task_domain": p["domain"], "benchmark": p["benchmark"],
                   "split": split, "source_index": p["index"], "source_file": relative,
                   "output_whitespace_words": len(raw.split()), "quality_flags": ";".join(qflags)}
            data[split].append(row)
            counter[split, p["benchmark"], p["domain"], family] += 1
            for flag in qflags:
                flag_counter[family, flag] += 1
                reviewed.append({"record_id": rid, "prompt_id": pid, "LLM_name": family, "flag": flag})
    for split, rows in data.items():
        rng.shuffle(rows)
        csv_write(output / f"{split}.csv", rows)
    csv_write(output / "split_manifest.csv", manifest)
    csv_write(output / "sample_counts.csv", [dict(zip(["split", "benchmark", "task_domain", "LLM_name", "rows"], (*k, v))) for k,v in sorted(counter.items())])
    csv_write(output / "excluded_groups.csv", excluded, ["source", "source_index", "reasons"])
    csv_write(output / "removed_exact_duplicates.csv", duplicates, ["removed_source", "removed_index", "retained_source", "retained_index"])
    csv_write(output / "group_links.csv", links, ["left_prompt_id", "right_prompt_id", "reason", "similarity"])
    csv_write(output / "review_flags.csv", reviewed, ["record_id", "prompt_id", "LLM_name", "flag"])
    # Validate task-critical invariants in memory before releasing files.
    assert len({p["prompt"] for p in pairs}) == n
    assert len({r["record_id"] for rows in data.values() for r in rows}) == n * 5
    group_splits = collections.defaultdict(set)
    for row in manifest:
        group_splits[row["group_id"]].add(row["split"])
    assert all(len(s) == 1 for s in group_splits.values())
    for rows in data.values():
        assert len(set(collections.Counter(r["LLM_name"] for r in rows).values())) == 1
    audit = {
        "dataset": "LLMRouterBench", "official_repository": "https://github.com/ynulihao/LLMRouterBench",
        "download_url": URL, "hf_revision": REVISION, "archive_sha256": archive_sha,
        "seed": SEED, "models": MODELS, "label_mapping": {i: f for i,f in enumerate(MODELS.values())},
        "source_files": provenance, "raw_rows_selected": sum(f["records"] for f in provenance),
        "complete_valid_prompts_before_dedup": pre_dedup, "excluded_groups": len(excluded),
        "removed_exact_duplicate_groups": len(duplicates), "prompts": n, "rows": n*5,
        "independent_split_groups": len(components), "group_links": len(links),
        "split_prompts": {s: len(rows)//5 for s,rows in data.items()},
        "split_rows": {s: len(rows) for s,rows in data.items()},
        "benchmark_prompts": dict(collections.Counter(p["benchmark"] for p in pairs)),
        "task_domain_prompts": dict(collections.Counter(p["domain"] for p in pairs)),
        "arena_domain_mapping_source_files": arena_map["source_files"],
        "quality_flag_counts": [{"model":m,"flag":f,"rows":v} for (m,f),v in sorted(flag_counter.items())],
        "near_duplicate_method": "NFKC/casefold/whitespace normalized origins; char_wb TFIDF(3,5), max_features120000, float32, cosine>=0.95, min80chars. Linked prompts kept together. Identical >=50word answers grouped too.",
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "split_method": "70/15/15 target, grouped and stratified by benchmark composition; integer/group rounding permitted",
        "limitations": ["Lexical grouping does not prove absence of semantic near duplicates or translations.",
                        "Short identical answers may recur across splits; different prompts can naturally have the same factual answer.",
                        "Public configurations do not establish identical provider internals or token budgets.",
                        "Text preserved including refusals, factual mistakes, repetition and possible cutoff.",
                        "No classifier has been trained; no accuracy/F1 results are implied.",
                        "No standalone data license found in the inspected official release; no new license is granted for third-party text."]}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ["raw_rows_selected","excluded_groups","removed_exact_duplicate_groups","prompts","rows","independent_split_groups","split_rows","benchmark_prompts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
