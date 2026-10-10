# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 — Midterm Project

**[Read the final report (PDF)](report.pdf)** · [Experiment code](Submission/run_project.py) · [Saved result tables](Submission/results/full/tables) · [Data provenance](Submission/DATA.md)

This project trains TextCNN and BiLSTM classifiers to identify which of five LLM
families produced a response: GPT, Claude, Gemini, Qwen, or DeepSeek. It studies
response-only attribution, the value of including the prompt, cross-domain
generalization, and the effects of formatting, identity keywords, and length.

The submitted results are already included. **Reading the report and checking
saved prediction metrics do not require retraining.** The final, author-edited
report is the root-level `report.pdf`; automatically generated drafts under
`Submission/results/` are supplementary outputs.

## Repository layout

| Path | Purpose |
|---|---|
| `report.pdf` | Final submitted report, with author and course information |
| `README.md` | Project overview and step-by-step reproduction instructions |
| `Submission/run_project.py` | Data checks, training, evaluation, and all four research questions |
| `Submission/attribution/` | Data processing, neural models, evaluation, and analysis |
| `Submission/config.json` | Default model, seed, training, and analysis settings |
| `Submission/prepare_data.py` | Reconstruct the dataset from a pinned external release |
| `Submission/DATA.md` | Exact source, model identifiers, preparation, and SHA-256 checks |
| `Submission/requirements.txt` | Pinned data-preparation dependencies |
| `Submission/requirements_training.txt` | Supported training/analysis dependencies |
| `Submission/tests/` | Small-fixture software checks |
| `Submission/verify_saved_results.py` | Recompute saved metrics with the Python standard library |
| `Submission/results/full/` | Completed experiments: predictions, metrics, histories, and comparisons |
| `Submission/report_assets/` | Figures used in the original report draft and their generation script |
| `Submission/evidence/` | Numerical audit, error examples, and preprocessing diagnostics |
| `Submission/provenance/` | Historical environment and packaging records |

The repository intentionally excludes raw/prepared text datasets, Python virtual
environments, trained weights, fitted vocabularies, and fitted feature-model
binaries. The existing results are sufficient for inspecting the experiments
and verifying saved prediction metrics. Full training requires the external
dataset; prediction on new text requires retraining a classifier first.

## Choose what to reproduce

| Goal | Follow these instructions | Needs source data? | Trains models? |
|---|---|---:|---:|
| Read the submitted work | Open `report.pdf` and the saved result tables | No | No |
| Check numerical results | Section A | No | No |
| Rebuild result tables/figures | Sections B and G | No | No |
| Check that the pipeline works | Sections B–E | Yes | Small smoke runs |
| Reproduce all four research questions | Sections B–D, then F | Yes | All full runs |
| Classify a new response | Section H, after training | During training | A trained classifier is required |

## A. Verify the existing results without retraining

Download the repository using GitHub's **Code → Download ZIP** and extract it,
or clone it with Git. Open the extracted/cloned repository folder in VS Code.
With Python installed, open a **PowerShell** terminal and run:

```powershell
cd Submission
py verify_saved_results.py
```

Expected success message:

```text
PASS: 23 saved runs verified; no training or external data used.
```

This recomputes test Accuracy and Macro-F1 for all 23 saved runs and checks
complete five-model prompt groups, probabilities, row counts, and input-only
controls. It verifies the saved evidence; it does not rerun neural inference
or independently witness the original training.

**All commands below assume your terminal is already inside `Submission/`.**
If opening a new terminal at the repository root, run `cd Submission` once.
If already inside `Submission`, do not run it again.

## B. Set up Python for training or analysis

Python 3.12 is recommended. On Windows, install Python and the Microsoft Python
extension for VS Code, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements_training.txt
```

These commands call the virtual environment's Python directly, so activating
the environment or changing PowerShell execution policy is unnecessary.
For VS Code's Run button, select **Python: Select Interpreter → Enter interpreter
path → Find**, then choose `Submission/.venv/Scripts/python.exe` inside the repository.
If that command is missing, install/enable the Microsoft Python extension and
reload VS Code; the terminal commands above remain usable independently.

`requirements.txt` pins the original preparation dependencies because approximate
near-duplicate grouping can depend on numeric-library versions.
`requirements_training.txt` lists supported training dependencies. The original
full training environment is recorded in
[`Submission/results/full/environment.json`](Submission/results/full/environment.json).
It differs from the preparation environment. The commands above reproduce the
workflow with compatible dependencies; identical numerical results across
library versions, hardware, and operating systems are not guaranteed.

The historical `formal_full_training_run=false` flag in
`Submission/provenance/original_smoke_environment.json` refers to the earlier
software smoke test, not to the completed full experiments.

On macOS/Linux, create the environment with `python3 -m venv .venv`, use
`.venv/bin/python` instead of `.\.venv\Scripts\python.exe`, and use `python3`
instead of `py`. See `DATA.md` for the matching download command.

## C. Obtain and prepare the data

The completed study uses **LLMRouterBench**. Data preparation reuses released
responses; it does not call LLM APIs or download the five source LLMs.
The frozen archive is approximately 1.28 GB. Details and exact model versions
are recorded in [`Submission/DATA.md`](Submission/DATA.md).

**Choose one data option:**

- **Previously prepared data available:** copy the original `prepared` folder,
  including `train.csv`, `validation.csv`, `test.csv`, and `audit.json`, into
  `Submission/`. Skip the download and preparation commands below.
- **Fresh reproduction:** download and prepare the pinned release as follows.
  The output folder `prepared` must be absent or empty.

```powershell
New-Item -ItemType Directory -Force raw | Out-Null
curl.exe -L --fail --retry 3 -o raw/bench-release.tar.gz "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/0e5af1b84bf73437a01a1849c0f1d2468baa93fc/bench-release.tar.gz"
Get-FileHash raw/bench-release.tar.gz -Algorithm SHA256
.\.venv\Scripts\python.exe prepare_data.py --archive raw/bench-release.tar.gz --output prepared
```

Expected archive SHA-256:

```text
b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1
```

The preparation script verifies the archive checksum itself. After preparation,
check all three dataset hashes:

```powershell
Get-FileHash prepared/train.csv, prepared/validation.csv, prepared/test.csv -Algorithm SHA256
```

| File | Expected SHA-256 |
|---|---|
| `train.csv` | `d5757433fef6410c001044537e9b7e8bc309f383b0af4e33ca73b70954b47761` |
| `validation.csv` | `8529a950fc6df390fa3a2bd22a279c2e63ec537cd445b6b85506a4457fcf70b6` |
| `test.csv` | `3f4fec288c3bfee985623b9f17516fc64f1c15f1036ff905a080452976357bd0` |

Investigate differences before treating a reconstruction as the same dataset.
Do not open and resave the generated CSVs in Excel before checking their hashes.
The documented archive was not downloaded and reconstructed again when this
submission package was assembled; the original run's hashes and preparation
code are retained for independent reproduction.

## D. Check the software and dataset

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe inspect_data.py
.\.venv\Scripts\python.exe run_project.py --stage check --output-dir results/reproduction_check
```

Expected dataset size: **57,565 responses from 11,513 prompts**. Each retained
prompt has one response from each of the five models. Train/validation/test
response counts are **40,295 / 8,635 / 8,635**. Complete prompt groups and detected
near-duplicate groups stay within one split.

`inspect_data.py` expects `prepared/audit.json` as well as the three CSVs.
The classifier uses text as its input; model identifiers and audit IDs are
metadata and must not be used as predictors.

## E. Optional small pipeline check

```powershell
.\.venv\Scripts\python.exe run_project.py --profile smoke --stage all --output-dir results/reproduction_smoke
```

This runs the pipeline using small samples, small models, and one epoch.
It is a software check, not a reproduction of the reported results.
Its outputs remain separate from both the submitted results and full reproduction.

## F. Reproduce all four research questions

Use the default `config.json` and this single command:

```powershell
.\.venv\Scripts\python.exe run_project.py --profile full --stage all --output-dir results/reproduction_full
```

This performs **22 neural runs plus one feature-based logistic regression run**,
then generates evaluation tables and plots. Main output-only and
input-plus-output conditions use seeds **42, 43, 44**. Input-only controls and
RQ3/RQ4 extras use seed **42**. Each neural run allows up to 15 epochs with early
stopping after three epochs without validation Macro-F1 improvement.

The recorded CPU experiment took approximately **97 hours across its neural
training runs**. This is a measured reference for the original machine, not a
runtime guarantee. Full reproduction can take several days on a CPU.

**Hardware choice:** the default `device: auto` uses CUDA when a CUDA-capable
PyTorch installation and compatible GPU are available, otherwise CPU. To require
CPU explicitly, add `--device cpu`. To require CUDA explicitly, add
`--device cuda`; it requires a working CUDA installation. Select the device
before starting and keep the settings consistent when resuming.

**Research-question choice:** to run in stages, use these commands *instead of*
the single `--stage all` command above. Keep the same output directory:

```powershell
.\.venv\Scripts\python.exe run_project.py --profile full --stage main --output-dir results/reproduction_full
.\.venv\Scripts\python.exe run_project.py --profile full --stage rq3 --output-dir results/reproduction_full
.\.venv\Scripts\python.exe run_project.py --profile full --stage rq4 --output-dir results/reproduction_full
.\.venv\Scripts\python.exe run_project.py --profile full --stage report --output-dir results/reproduction_full
```

| Stage | Purpose |
|---|---|
| `main` | RQ1 and RQ2: both architectures and all three text-input conditions |
| `rq3` | Cross-domain and matched mixed-domain comparisons for coding and factual QA |
| `rq4` | Feature baseline, formatting removal, keyword masking, and length controls |
| `report` | Summarize existing outputs; no training |
| `all` | Run all experiment stages and generate summaries |

Do not shorten the full run by changing epochs, seeds, or sample sizes and then
claim it reproduces the original configuration. If you change settings for a
new experiment, document the differences and use a new output directory.

Repeating the same interrupted training command resumes compatible local
checkpoints from the last completed epoch. The archived `results/full` folder
omits weights and is not a checkpoint bundle. **Always keep the explicit
`--output-dir results/reproduction_full` argument when reproducing:** without a
custom output directory, the full profile targets `results/full` by default.
Changing code, configuration, or data requires a new output directory.

## G. Find outputs or rebuild tables without training

Submitted outputs are in `Submission/results/full/`. New outputs from Section F
are in `Submission/results/reproduction_full/`.

| File or directory relative to a results folder | What to inspect |
|---|---|
| `tables/main_results.csv` | RQ1/RQ2 Accuracy and Macro-F1, aggregated over seeds |
| `tables/rq2_paired_deltas.csv` | Paired information-condition comparisons |
| `tables/rq3_paired_deltas.csv` | Cross-domain comparisons |
| `tables/rq4_paired_deltas.csv` | Ablation comparisons |
| `tables/all_run_metrics.csv` | Metrics for individual runs |
| `tables/research_question_status.csv` | Coverage of the research questions |
| `figures/` | Automatically generated charts |
| `runs/` | Neural run predictions, metrics, histories, and confusion matrices |
| `rq4_features/` | Feature-based logistic regression results |
| `experiment_index.json` | Index of completed experiments |

To rebuild the saved tables and generated summaries **without the source data
or retraining**, install Section B's dependencies and work on a copy:

```powershell
Copy-Item -Recurse results/full results/reproduction_summary
.\.venv\Scripts\python.exe run_project.py --stage report --output-dir results/reproduction_summary
```

Run the copy command once into a new destination; do not repeatedly copy into
an existing destination. The original `results/full` evidence stays unchanged.
The generated `report_draft.html` and `results_summary.pdf` compile results;
they do not regenerate or replace the manually edited root `report.pdf`.

Large selection manifests use `.csv.gz`; pandas can read these directly.
`Submission/PACKAGE_CONTENTS.json` and records under `provenance/` describe the
original compact results package before the final repository layout and report
edits. `Submission/SOURCE_SHA256SUMS.txt` covers the included source/configuration.
The old package-wide `DELIVERY_SHA256SUMS.txt` should not be used as a checksum
manifest for a subsequently edited submission.

## H. Optional prediction on a new response

After completing training in Section F, create an `answer.txt` file inside
`Submission/` and run:

```powershell
.\.venv\Scripts\python.exe predict_text.py --run-dir results/reproduction_full/runs/main_cnn_output_only_s42 --output-file answer.txt
```

Choose the run directory for the trained classifier you want to use. A
classifier trained with input-plus-output also needs
`--prompt-file question.txt`. This requires the newly generated `best.pt` and
`vocabulary.json`; the submitted evidence folder deliberately omits them.
The classifier chooses among five known classes and does not detect an unknown
sixth model. Its softmax scores are not calibrated authorship probabilities.

## Research design

The dataset has 11,513 retained prompts, five responses per prompt, and 57,565
responses in total. The splits contain 40,295 training, 8,635 validation, and
8,635 test responses. All five responses to a prompt stay together; linked
near-duplicate prompts also remain in one split. The supplied grouping does
not imply that every possible semantic paraphrase has been detected.

1. **RQ1 — Response-only attribution.** Compare CNN and BiLSTM using the first
   512 response tokens. Each architecture is trained with seeds 42, 43, and 44.
2. **RQ2 — Information available to the classifier.** Compare input-only,
   output-only, and input-plus-output. Input-only is a negative control using
   seed 42; the other conditions use three seeds. Input-plus-output preserves
   the same 512-token response prefix and adds up to 256 prompt tokens and a
   separator. Deterministic input-only accuracy is exactly 20% because the
   same model-visible prompt appears once with every label; its Macro-F1 need
   not be 20%.
3. **RQ3 — Cross-domain generalization.** For coding and factual QA, compare a
   CNN trained without the target domain with a mixed-domain CNN matched in
   training-prompt count. Both conditions use the same non-target validation
   set and the same target-domain test records. These additional experiments
   use seed 42.
4. **RQ4 — Predictive text features and ablations.** Examine training-set text
   features, fit an interpretable logistic regression baseline, and compare
   removal of formatting, masking of model/provider keywords, and fixed-length
   responses. The length threshold was selected from training data and equals
   256 tokens. The length comparison uses the same eligible records in both
   conditions; it is conditional on sufficiently long responses. Extra neural
   experiments use seed 42.

The vocabulary is fitted on allowed training text only. Neural models use
Adam, cross-entropy, gradient clipping, and early stopping on validation
Macro-F1. Test results are computed after selecting the best validation epoch.
Bootstrap intervals resample `group_id` clusters rather than individual
responses. Across-seed standard deviations and test-sample bootstrap
uncertainty answer different questions and are reported separately.

## Saved main results

The following values come from `results/full/tables/main_results.csv`.
Mean and sample standard deviation are expressed as percentages or percentage
points, respectively; a dash indicates a single run.

| Classifier | Input | Seeds | Accuracy, mean ± SD | Macro-F1, mean ± SD |
|---|---|---:|---:|---:|
| CNN | Input only | 1 | 20.00 | 18.32 |
| CNN | Output only | 3 | 86.71 ± 0.51 | 86.67 ± 0.49 |
| CNN | Input + output | 3 | 86.50 ± 0.16 | 86.48 ± 0.13 |
| BiLSTM | Input only | 1 | 20.00 | 15.24 |
| BiLSTM | Output only | 3 | 85.67 ± 0.43 | 85.71 ± 0.39 |
| BiLSTM | Input + output | 3 | 84.77 ± 0.54 | 84.77 ± 0.48 |

These are attribution results for the particular released model versions and
collection settings. They do not establish universal model fingerprints or
causal explanations of model behavior. RQ3 and RQ4 are single-seed additional
analyses; their test bootstrap intervals do not measure training-seed variation.

## Report and provenance notes

The final report is [`report.pdf`](report.pdf) at the repository root.
`Submission/report_tools/build_report.py` is an optional helper for rebuilding
an earlier Word draft; it contains the original author placeholder and does
not reproduce subsequent manual edits to the final PDF. It is not needed for
training, evaluation, or numerical reproduction. The generic draft flag in
`Submission/results/full/report_status.json` likewise describes an automated
scaffold, not missing or failed experiments.

The original saved results include 23 prediction sets. Numerical values and
trained source code were preserved when making the compact package; selected
large manifests were losslessly compressed, fitted binaries were omitted, and
machine-specific environment/comparison paths were made relative.

## Data reference

Li, H., et al. (2026). *LLMRouterBench: A Massive Benchmark and Unified Framework
for LLM Routing*. [arXiv:2601.07206](https://arxiv.org/abs/2601.07206).

- [Official source repository](https://github.com/ynulihao/LLMRouterBench)
- [Official dataset](https://huggingface.co/datasets/NPULH/LLMRouterBench)
- [Frozen release, model identifiers, and reconstruction details](Submission/DATA.md)
