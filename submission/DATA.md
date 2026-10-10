# Data provenance and reconstruction

This submission contains code and derived evaluation outputs. **It does not
bundle either the external raw archive or the prepared prompt/response CSVs.**
Training from a fresh clone requires the download and preparation steps below.

## Frozen source

- Dataset: [NPULH/LLMRouterBench](https://huggingface.co/datasets/NPULH/LLMRouterBench).
- Source repository: <https://github.com/ynulihao/LLMRouterBench>.
- Frozen dataset revision: `0e5af1b84bf73437a01a1849c0f1d2468baa93fc`.
- Archive filename: `bench-release.tar.gz`.
- Archive size recorded during preparation: **1,283,503,080 bytes**.
- SHA-256: `b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1`.
- [Direct frozen archive download](https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/0e5af1b84bf73437a01a1849c0f1d2468baa93fc/bench-release.tar.gz).

The preparation script uses existing responses; it never requests new responses
from a model. The five source models are:

| Label | Family | Exact released model identifier |
|---:|---|---|
| 0 | GPT | `gpt-5-chat` |
| 1 | Claude | `claude-sonnet-4` |
| 2 | Gemini | `gemini-2.5-flash` |
| 3 | Qwen | `qwen3-235b-a22b-2507` |
| 4 | DeepSeek | `deepseek-v3-0324` |

The underlying benchmark sources are AIME, LiveMathBench, LiveCodeBench, GPQA,
HLE, MMLU-Pro, SimpleQA, and ArenaHard. Coarse task domains include mathematics,
coding, knowledge/reasoning, factual QA, and creative writing. ArenaHard domain
assignments and their source hashes are frozen in `arena_task_domains.json`.
Benchmark origin is not a perfect semantic task taxonomy.

## Download and prepare

Run from the `Submission/` folder, after installing the dependencies in the
[root README](../README.md). From the repository root, first run `cd Submission`.
These commands are for Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force raw | Out-Null
curl.exe -L --fail --retry 3 -o raw/bench-release.tar.gz "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/0e5af1b84bf73437a01a1849c0f1d2468baa93fc/bench-release.tar.gz"
Get-FileHash raw/bench-release.tar.gz -Algorithm SHA256
.\.venv\Scripts\python.exe prepare_data.py --archive raw/bench-release.tar.gz --output prepared
.\.venv\Scripts\python.exe inspect_data.py
```

`prepare_data.py` verifies the archive's SHA-256 itself before reading the
needed JSON members. **The `--output` directory must be absent or empty.** If
`prepared/` already contains the three previously prepared files, use them
directly; do not overwrite them. To compare a fresh reconstruction, choose
`--output prepared_rebuilt` and subsequently use
`run_project.py --data-dir prepared_rebuilt ...`. Preparation also writes
`audit.json` and provenance/selection tables. `inspect_data.py` expects that
`audit.json` alongside the CSVs; the training program reads and checks the
three CSVs directly.

Equivalent download/preparation commands on macOS/Linux are:

```bash
mkdir -p raw
curl -L --fail --retry 3 -o raw/bench-release.tar.gz 'https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/0e5af1b84bf73437a01a1849c0f1d2468baa93fc/bench-release.tar.gz'
.venv/bin/python prepare_data.py --archive raw/bench-release.tar.gz --output prepared
.venv/bin/python inspect_data.py
```

Use the pinned versions in `requirements.txt` for the preparation step. The
script performs approximate near-duplicate grouping using numeric libraries;
the retained dataset hashes provide a stronger reproduction check than sample
counts alone. The `--selected` option is intended for a previously extracted
cache; the archive-based command above is the documented reconstruction path.

## Retained data and split checks

The main record fields are `record_id`, `prompt_id`, `group_id`, `label`,
`LLM_name`, `model_id`, `LLM_Input`, `LLM_output`, `task_domain`, `benchmark`, and
`split`. The classifiers read text and the target label; IDs and model metadata
are retained for auditing and must not be used as predictors.

The procedure retains complete groups of five models with byte-identical
prompts, excludes missing/empty responses and identified API-failure strings,
removes normalized exact duplicate questions, and groups detected near
duplicates before splitting. Ordinary short responses, refusals, and incorrect
answers are not automatically removed as API failures.

| Split | Prompts | Responses | Responses per family |
|---|---:|---:|---:|
| Training | 8,059 | 40,295 | 8,059 |
| Validation | 1,727 | 8,635 | 1,727 |
| Test | 1,727 | 8,635 | 1,727 |
| Total | 11,513 | 57,565 | 11,513 |

Expected SHA-256 values for the exact CSV files used in the recorded full run:

```text
d5757433fef6410c001044537e9b7e8bc309f383b0af4e33ca73b70954b47761  train.csv
8529a950fc6df390fa3a2bd22a279c2e63ec537cd445b6b85506a4457fcf70b6  validation.csv
3f4fec288c3bfee985623b9f17516fc64f1c15f1036ff905a080452976357bd0  test.csv
```

These also appear in `results/full/data_hashes.json`. Inspect the hashes with:

```powershell
Get-FileHash prepared/train.csv, prepared/validation.csv, prepared/test.csv -Algorithm SHA256
```

Different line endings or text encodings can change file hashes even when
parsed records are unchanged; preserve the generated files as written. If
reconstruction differs, investigate before treating it as the same experiment.
Do not rename unrelated data files to make them appear to match this dataset.

The archive was not downloaded or rebuilt again during preparation of this
compact submission package. The download URL, checksum, flags, and required
support file were checked against the bundled preparation code; the original
full-run data audit and hashes are retained as evidence.

## Scope and rights

The study attributes outputs from these particular released model versions
under the benchmark's collection conditions. Model family, provider/deployment,
system prompts, decoding implementation, and other generation settings can be
confounded. Results therefore should not be interpreted as invariant signatures
of every release in a model family.

The inspected release did not provide a separate unambiguous license for all
underlying prompt/response text. This package grants no new rights to third-party
data. External source links and preparation code are supplied instead of
redistributing the text corpus. Retain the benchmark and original task-source
attributions when reusing data.
