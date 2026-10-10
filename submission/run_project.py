"""Run the complete CMPSC 448 midterm pipeline from VS Code / PowerShell.

Examples (run inside the midterm directory):
    python run_project.py --stage check
    python run_project.py --profile smoke --stage all
    python run_project.py --profile full --stage all

The default data directory is ./prepared next to THIS file, independent of the
terminal working directory. No paid API or external LLM requests are made.
All training/evaluation is local. Smoke outputs never enter full-run tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent


def write_json(path, data):
    """Atomic write so an interruption cannot leave a half-written manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=BASE / "prepared")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="A new directory if you change the experiment settings")
    p.add_argument("--config", type=Path, default=BASE / "config.json")
    p.add_argument("--profile", choices=["smoke", "full"], default="full")
    p.add_argument("--stage", choices=["check", "main", "rq3", "rq4", "report", "all"], default="all")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Seeds for main output-only and input+output runs")
    p.add_argument("--extra-seeds", type=int, nargs="+", default=None,
                   help="Seeds for RQ3 and RQ4 (default 42; use 42 43 44 for repetitions)")
    p.add_argument("--no-resume", action="store_true", help="Do not resume; use a NEW output directory")
    return p.parse_args()


def effective_config(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg["profile"] = args.profile
    if args.profile == "smoke":
        cfg.update(epochs=1, patience=1, vocabulary_size=2000, embedding_dim=16,
                   cnn_filters=8, lstm_hidden=12, batch_size=16,
                   input_max_tokens=48, output_max_tokens=96,
                   seeds=[42], extra_seeds=[42], bootstrap_reps=40,
                   length_candidates=[16, 32, 64], length_min_train_prompts=5,
                   smoke_group_limits={"train": 60, "validation": 30, "test": 30})
    for argname, key in [("epochs", "epochs"), ("batch_size", "batch_size"),
                         ("threads", "threads"), ("device", "device"),
                         ("seeds", "seeds"), ("extra_seeds", "extra_seeds")]:
        value = getattr(args, argname)
        if value is not None:
            cfg[key] = value
    for key in ["epochs", "batch_size", "threads", "input_max_tokens", "output_max_tokens"]:
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not cfg["seeds"] or not cfg["extra_seeds"]:
        raise ValueError("Seed lists cannot be empty")
    cfg["seeds"] = list(dict.fromkeys(cfg["seeds"]))
    cfg["extra_seeds"] = list(dict.fromkeys(cfg["extra_seeds"]))
    return cfg


def prompt_count(frame):
    return int(frame["prompt_id"].nunique())


def exclude_domain_groups(frame, domain):
    """Exclude the WHOLE connected component whenever it contains target data."""
    blocked = set(frame.loc[frame.task_domain.eq(domain), "group_id"])
    return frame.loc[~frame.group_id.isin(blocked)].copy().reset_index(drop=True)


def matched_training_sample(frame, n_prompts, domain, seed):
    """Exactly match prompt count, keep groups intact and include target domain.

    Most connected components contain one prompt. Randomly sample larger groups,
    then fill the exact residual with singleton groups. This preserves pairing
    without duplicated/oversampled rows. Allocation uses metadata only.
    """
    import numpy as np
    sizes = frame.groupby("group_id", sort=True).prompt_id.nunique()
    target_groups = set(frame.loc[frame.task_domain.eq(domain), "group_id"])
    rng = np.random.default_rng(seed)
    singleton = sizes[sizes.eq(1)].index.to_numpy(dtype=object)
    target_singletons = sorted(set(singleton) & target_groups)
    if not target_singletons:
        raise ValueError(f"No singleton target group for matched training: {domain}")
    if not 1 <= n_prompts <= prompt_count(frame):
        raise ValueError("Invalid requested matched sample size")
    forced = target_singletons[int(rng.integers(len(target_singletons)))]
    chosen = {forced}
    remaining = n_prompts - 1
    fraction = n_prompts / prompt_count(frame)
    larger = list(sizes[sizes.gt(1)].items())
    rng.shuffle(larger)
    for gid, weight in larger:
        if rng.random() < fraction and weight <= remaining:
            chosen.add(gid)
            remaining -= int(weight)
    pool = np.array(sorted(set(singleton) - chosen), dtype=object)
    if len(pool) < remaining:
        # This dataset has ample singleton components. Fail transparently if a
        # replacement dataset violates that assumption; never split a group.
        raise ValueError("Not enough singleton groups for exact size matching")
    chosen.update(rng.choice(pool, size=remaining, replace=False).tolist())
    selected = frame.loc[frame.group_id.isin(chosen)].copy().reset_index(drop=True)
    assert prompt_count(selected) == n_prompts
    assert selected.task_domain.eq(domain).any()
    assert selected.groupby("prompt_id").size().eq(5).all()
    return selected


def save_selection(path, frames):
    import pandas as pd
    cols = ["record_id", "prompt_id", "group_id", "label", "task_domain", "benchmark"]
    records = []
    for role, frame in frames.items():
        piece = frame[cols].copy()
        piece.insert(0, "role", role)
        records.append(piece)
    pd.concat(records, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")


class Pipeline:
    def __init__(self, splits, cfg, root, resume=True):
        self.splits, self.cfg, self.root, self.resume = splits, cfg, root, resume
        self.index_path = root / "experiment_index.json"
        self.index = json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.exists() else []

    def run(self, name, rq, frames=None, architecture="cnn", mode="output_only",
            seed=42, transform="none", length_k=None, condition=None, target=None,
            reference=None):
        from attribution.engine import run_experiment
        frames = self.splits if frames is None else frames
        relative = "runs/" + name
        entry = {"run_id": name, "rq": rq, "run_dir": relative,
                 "architecture": architecture, "mode": mode, "seed": seed,
                 "transform": transform, "condition": condition or transform,
                 "target": target, "length_k": length_k,
                 "reference_run_dir": reference, "profile": self.cfg["profile"]}
        found = next((i for i, x in enumerate(self.index) if x["run_dir"] == relative), None)
        if found is None:
            self.index.append(entry)
        else:
            self.index[found] = entry
        write_json(self.index_path, self.index)
        print(f"\n{'='*65}\n{name}: {len(frames['train']):,} train / "
              f"{len(frames['validation']):,} validation / {len(frames['test']):,} test rows", flush=True)
        metrics = run_experiment(frames["train"], frames["validation"], frames["test"],
                                 self.cfg, self.root / relative, architecture=architecture,
                                 mode=mode, seed=seed, transform=transform,
                                 length_k=length_k, resume=self.resume)
        if mode == "input_only" and abs(float(metrics["accuracy"]) - .2) > 1e-8:
            raise AssertionError("Input-only accuracy must equal 20% on complete five-way paired data. Check evaluation/pairing.")
        return relative

    def raw_cnn(self, seed):
        return self.run(f"main_cnn_output_only_s{seed}", "main", seed=seed, condition="output_only")

    def main_experiments(self):
        # Output-only conditions answer RQ1 AND form part of the RQ2 comparison.
        for architecture in ["cnn", "bilstm"]:
            for mode in ["input_only", "output_only", "input_and_output"]:
                seeds = self.cfg["seeds"][:1] if mode == "input_only" else self.cfg["seeds"]
                for seed in seeds:
                    self.run(f"main_{architecture}_{mode}_s{seed}", "main",
                             architecture=architecture, mode=mode, seed=seed, condition=mode,
                             reference=f"runs/main_{architecture}_output_only_s{seed}"
                             if mode == "input_and_output" else None)

    def cross_domain(self):
        folder = self.root / "rq3_design"
        folder.mkdir(exist_ok=True)
        for target in self.cfg["rq3_domains"]:
            ood_train = exclude_domain_groups(self.splits["train"], target)
            common_val = exclude_domain_groups(self.splits["validation"], target)
            target_test = self.splits["test"].loc[self.splits["test"].task_domain.eq(target)].copy()
            if min(len(ood_train), len(common_val), len(target_test)) == 0:
                raise ValueError(f"RQ3 {target}: an empty partition; choose a larger smoke sample/new target before analysis")
            for seed in self.cfg["extra_seeds"]:
                mixed_train = matched_training_sample(self.splits["train"], prompt_count(ood_train), target, seed)
                assert len(mixed_train) == len(ood_train)
                assert not ood_train.task_domain.eq(target).any()
                assert not common_val.task_domain.eq(target).any()
                save_selection(folder / f"{target}_s{seed}_selection.csv",
                               {"ood_train": ood_train, "mixed_train": mixed_train,
                                "common_validation": common_val, "target_test": target_test})
                write_json(folder / f"{target}_s{seed}_design.json", {
                    "target_domain": target, "seed": seed,
                    "ood_train_prompts": prompt_count(ood_train),
                    "mixed_train_prompts": prompt_count(mixed_train),
                    "mixed_target_prompts": prompt_count(mixed_train.loc[mixed_train.task_domain.eq(target)]),
                    "common_validation_prompts": prompt_count(common_val),
                    "target_test_prompts": prompt_count(target_test),
                    "validation_rule": "same non-target connected groups in both conditions",
                    "vocabulary_rule": "fitted separately using each permitted training set only"})
                frames = {"train": mixed_train, "validation": common_val, "test": target_test}
                reference = self.run(f"rq3_{target}_mixed_s{seed}", "rq3", frames, seed=seed,
                                     condition="matched_mixed", target=target)
                frames = {"train": ood_train, "validation": common_val, "test": target_test}
                self.run(f"rq3_{target}_heldout_s{seed}", "rq3", frames, seed=seed,
                         condition="target_heldout", target=target, reference=reference)

    def feature_experiments(self):
        import numpy as np
        from attribution.data import token_lengths
        from attribution.analysis import run_feature_baseline
        design_dir = self.root / "rq4_design"
        design_dir.mkdir(exist_ok=True)
        print("RQ4: choosing the fixed token length using TRAINING data only...", flush=True)
        train = self.splits["train"]
        lengths = token_lengths(train, max_tokens=self.cfg["output_max_tokens"])
        temp = train[["prompt_id", "group_id"]].copy()
        temp["length"] = lengths
        group_min = temp.groupby("group_id").length.min()
        candidates = []
        for k in self.cfg["length_candidates"]:
            if not 0 < k < self.cfg["output_max_tokens"]:
                continue
            ids = set(group_min[group_min.ge(k)].index)
            n = prompt_count(train.loc[train.group_id.isin(ids)])
            candidates.append({"k_tokens": k, "training_prompts": n,
                               "training_retention": n / prompt_count(train)})
        eligible = [x for x in candidates if x["training_retention"] >= self.cfg["length_min_train_retention"]
                    and x["training_prompts"] >= self.cfg["length_min_train_prompts"]]
        if not eligible:
            raise ValueError("No fixed-length candidate meets predeclared training-only retention. See config; do not choose by test accuracy.")
        k = max(x["k_tokens"] for x in eligible)
        length_frames = {}
        retention = {}
        for split, frame in self.splits.items():
            values = lengths if split == "train" else token_lengths(frame, max_tokens=self.cfg["output_max_tokens"])
            counts = frame[["group_id"]].copy()
            counts["length"] = values
            minima = counts.groupby("group_id").length.min()
            ids = set(minima[minima.ge(k)].index)
            selected = frame.loc[frame.group_id.isin(ids)].copy().reset_index(drop=True)
            if selected.empty:
                raise ValueError(f"No {split} groups qualify for preselected K={k}; report this limitation, do not tune K on test results")
            assert selected.groupby("prompt_id").size().eq(5).all()
            length_frames[split] = selected
            retention[split] = {"rows": len(selected), "prompts": prompt_count(selected),
                                "fraction": prompt_count(selected) / prompt_count(frame)}
        write_json(design_dir / "length_selection.json", {
            "selection_uses": "training text lengths only", "unit": "actual tokenizer tokens",
            "candidates": candidates, "chosen_k": k, "retention": retention,
            "eligibility": "every answer in each retained connected group has at least K tokens",
            "comparison": "same records and splits: original up-to-512-token prefix vs exact first K tokens",
            "limitation": "conditional on long-enough answers; cannot generalize its result to all short answers"})
        save_selection(design_dir / "length_subset_selection.csv", length_frames)
        for seed in self.cfg["extra_seeds"]:
            raw = self.raw_cnn(seed)
            self.run(f"rq4_no_format_s{seed}", "rq4", seed=seed, transform="no_format",
                     condition="no_format", reference=raw)
            self.run(f"rq4_keyword_mask_s{seed}", "rq4", seed=seed, transform="mask_identity",
                     condition="keyword_mask", reference=raw)
            length_raw = self.run(f"rq4_length_raw_s{seed}", "rq4", frames=length_frames,
                                  seed=seed, condition="length_subset_raw", target=f"K={k}")
            self.run(f"rq4_fixed_length_s{seed}", "rq4", frames=length_frames,
                     seed=seed, transform="fixed_length", length_k=k,
                     condition="fixed_token_length", target=f"K={k}", reference=length_raw)
        print("RQ4: training the interpretable feature-only baseline...", flush=True)
        run_feature_baseline(self.splits, self.root / "rq4_features", self.cfg)
        entry = {"run_id": "rq4_features", "rq": "rq4", "run_dir": "rq4_features",
                 "architecture": "feature_logistic", "mode": "output_features",
                 "condition": "linguistic_features", "seed": 42,
                 "target": None, "profile": self.cfg["profile"]}
        self.index = [e for e in self.index if e["run_dir"] != "rq4_features"] + [entry]
        write_json(self.index_path, self.index)


def main():
    args = parse_args()
    root = (args.output_dir or BASE / "results" / args.profile).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import torch
        import matplotlib
    except ImportError as error:
        raise SystemExit(f"Missing dependency: {error}. Run: python -m pip install -r requirements_training.txt") from error
    from attribution.analysis import describe_data, generate_report
    from attribution.data import load_splits, audit_splits, sample_groups
    if args.stage == "report":
        saved = root / "effective_config.json"
        if not saved.exists():
            raise ValueError(f"No experiment configuration at {saved}. Train first or use --output-dir.")
        cfg = json.loads(saved.read_text(encoding="utf-8"))
        generate_report(root, cfg)
        print(f"Report refreshed: {root}", flush=True)
        return
    cfg = effective_config(args)
    saved = root / "effective_config.json"
    if saved.exists():
        previous = json.loads(saved.read_text(encoding="utf-8"))
        if previous != cfg:
            raise ValueError("This output directory already uses different settings. Reuse identical arguments to resume, or choose --output-dir results/new_experiment.")
    else:
        write_json(saved, cfg)
    # Protect a result directory against unnoticed changes to the source data.
    data_dir = args.data_dir.resolve()
    for name in ["train.csv", "validation.csv", "test.csv"]:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"Missing {data_dir/name}. Put this code next to your prepared folder, or pass --data-dir.")
    hashes = {name: hash_file(data_dir/name) for name in ["train.csv", "validation.csv", "test.csv"]}
    hash_path = root / "data_hashes.json"
    if hash_path.exists() and json.loads(hash_path.read_text()) != hashes:
        raise ValueError("CSV contents changed. Use a new output directory to avoid mixing experiments.")
    write_json(hash_path, hashes)
    write_json(root / "environment.json", {"python": sys.version, "platform": platform.platform(),
               "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
               "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__,
               "cuda_available": torch.cuda.is_available(),
               "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "data_directory": str(data_dir)})
    print(f"Profile: {args.profile.upper()} | Data: {data_dir}\nResults: {root}", flush=True)
    if args.profile == "smoke":
        print("SMOKE = small data, one epoch, small networks. These scores are NOT final research results.", flush=True)
    print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}", flush=True)
    print("Reading and checking all three data splits...", flush=True)
    splits = load_splits(data_dir)
    audit_splits(splits, root / "data_audit_full")
    if args.profile == "smoke":
        splits = {s: sample_groups(df, cfg["smoke_group_limits"][s], cfg["data_seed"] + i)
                  for i, (s, df) in enumerate(splits.items())}
        audit_splits(splits, root / "data_audit_used")
    else:
        # A separate used-data audit makes smoke/full report handling explicit.
        (root / "data_audit_used").mkdir(exist_ok=True)
        import shutil
        for file in (root / "data_audit_full").glob("*"):
            if file.is_file():
                shutil.copy2(file, root / "data_audit_used" / file.name)
    print("Data checks passed. Exporting data descriptions...", flush=True)
    describe_data(splits, root / "descriptive", cfg)
    if args.stage == "check":
        print(f"CHECK complete. Next: python run_project.py --profile smoke --stage all\nSee {root}", flush=True)
        return
    pipeline = Pipeline(splits, cfg, root, resume=not args.no_resume)
    if args.stage in ["main", "all"]:
        pipeline.main_experiments()
    if args.stage in ["rq3", "all"]:
        pipeline.cross_domain()
    if args.stage in ["rq4", "all"]:
        pipeline.feature_experiments()
    print("Building comparison tables, figures and report draft...", flush=True)
    generate_report(root, cfg)
    write_json(root / "last_completed_stage.json", {"stage": args.stage, "profile": args.profile})
    print(f"\nCOMPLETED: {args.stage} ({args.profile}). Results are in {root}", flush=True)


if __name__ == "__main__":
    # Required for Windows DataLoader process startup and VS Code execution.
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed epochs are saved. Run the SAME command to resume.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        print("\nRun stopped; no missing results have been invented. Fix the error and rerun the same command. "
              "If changing settings, use a new --output-dir.", file=sys.stderr)
        sys.exit(1)
