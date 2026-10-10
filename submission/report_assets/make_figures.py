"""Reproduce report figures from the uploaded full-run results (no training)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results' / 'full'
OUT = Path(__file__).resolve().parent
metrics = pd.read_csv(DATA / 'tables/all_run_metrics.csv').set_index('run_id')
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#667085', 'axes.linewidth': .7,
    'text.color': '#202938', 'axes.labelcolor': '#202938',
    'xtick.color': '#344054', 'ytick.color': '#344054',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
})
BLUE, ORANGE, GRAY = '#2369A0', '#CC7728', '#8492A5'
captions = {}

def finish(fig, name, caption):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=.10)
    captions[name] = caption
    plt.close(fig)

def score_axis(ax):
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.grid(axis='y', color='#E5E9EF', linewidth=.65)
    ax.set_axisbelow(True)
    ax.set_ylabel('Test macro-F1')

# 1. Architecture and input information: dots show the actual seed runs.
fig, ax = plt.subplots(figsize=(7.4, 4.1))
modes = ['input_only', 'output_only', 'input_and_output']
for j, (arch, title, color) in enumerate([('cnn', 'CNN', BLUE), ('bilstm', 'BiLSTM', ORANGE)]):
    xx = np.arange(3) + (j - .5) * .34
    for i, mode in enumerate(modes):
        d = metrics.loc[(metrics.rq == 'main') & (metrics.architecture == arch) & (metrics['mode'] == mode)].sort_values('seed')
        yy = d.macro_f1.to_numpy()
        ax.bar(xx[i], yy.mean(), width=.30, color=color, alpha=.86, zorder=2)
        jitter = np.linspace(-.075, .075, len(yy)) if len(yy) > 1 else np.array([0.])
        ax.scatter(xx[i] + jitter, yy, s=22, color='#172231', edgecolor='white', linewidth=.55, zorder=4)
        ax.text(xx[i], yy.max() + .022, f'{yy.mean():.3f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(np.arange(3), ['Input only\n1 seed per architecture', 'Output only\n3 seeds per architecture', 'Input + output\n3 seeds per architecture'])
score_axis(ax)
ax.legend(handles=[Patch(facecolor=BLUE, label='CNN'), Patch(facecolor=ORANGE, label='BiLSTM'), Line2D([], [], marker='o', linestyle='', color='#172231', markersize=4, label='Individual seed')],
          loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=3, frameon=False)
fig.subplots_adjust(bottom=.18, top=.87, left=.10, right=.99)
finish(fig, 'fig01_main_macro_f1', 'Test macro-F1 by architecture and input condition. Bars show seed means; dots show individual fitted runs. Input-only uses seed 42; output-only and input-plus-output use seeds 42, 43, and 44. Every condition is evaluated on the same 8,635 answers. No uncertainty bar is reported for the single-seed input-only condition. Input-only accuracy is exactly 0.20 for both architectures; macro-F1 is not constrained to equal 0.20.')

# 2. Cross-domain controls use matched training sizes and identical target tests.
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.05), sharey=True)
for ax, domain, name in zip(axes, ['coding', 'factual_qa'], ['Coding', 'Factual QA']):
    design = json.loads((DATA / 'rq3_design' / f'{domain}_s42_design.json').read_text())
    ids = [f'rq3_{domain}_mixed_s42', f'rq3_{domain}_heldout_s42']
    d = metrics.loc[ids]
    vals = d.macro_f1.to_numpy()
    errors = np.vstack([vals - d.macro_f1_ci_low.to_numpy(), d.macro_f1_ci_high.to_numpy() - vals])
    ax.bar([0, 1], vals, width=.60, color=[BLUE, ORANGE], yerr=errors,
           error_kw={'elinewidth': .9, 'capsize': 3, 'capthick': .9}, zorder=2)
    for i, val in enumerate(vals):
        ax.text(i, float(d.macro_f1_ci_high.iloc[i]) + .026, f'{val:.3f}', ha='center', fontsize=10)
    ax.set_xticks([0, 1], ['Matched mixed', 'Target held out'])
    ax.set_title(name, pad=10, fontweight='bold')
    score_axis(ax)
    ax.set_xlabel(f"Training: {design['ood_train_prompts']:,} prompts per condition\nTarget test: {design['target_test_prompts']:,} prompts", fontsize=9, labelpad=10)
axes[1].set_ylabel('')
fig.subplots_adjust(bottom=.24, top=.88, left=.09, right=.99, wspace=.12)
finish(fig, 'fig02_cross_domain', 'Cross-domain transfer for the CNN output-only classifier (seed 42). Matched-mixed training contains the target domain; target-held-out training excludes it. Within each panel, both conditions use the same number of training prompts, the same non-target validation set, and the same target-domain test set. Each prompt contributes five answers. Error bars are 95% group-bootstrap intervals conditional on each fitted model (500 resamples); they do not capture variation across training seeds. Paired macro-F1 drops are 0.358 for coding and 0.398 for factual QA.')

# 3. Keep the restricted length sample separate from the full-sample ablations.
fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.2), sharey=True, gridspec_kw={'width_ratios': [1.45, 1]})
panels = [
    (['main_cnn_output_only_s42', 'rq4_no_format_s42', 'rq4_keyword_mask_s42'], ['Original\nprefix', 'Formatting\nremoved', 'Identity words\nmasked'], 'A. Full test set', '8,635 answers; same records', [BLUE, ORANGE, GRAY]),
    (['rq4_length_raw_s42', 'rq4_fixed_length_s42'], ['Original prefix\n(up to 512 tokens)', 'Fixed prefix\n(256 tokens)'], 'B. Long-answer subset', '3,520 answers; same records within this panel', [BLUE, ORANGE]),
]
for ax, (ids, names, title, desc, colors) in zip(axes, panels):
    d = metrics.loc[ids]
    values = d.macro_f1.to_numpy()
    errors = np.vstack([values - d.macro_f1_ci_low.to_numpy(), d.macro_f1_ci_high.to_numpy() - values])
    ax.bar(np.arange(len(ids)), values, color=colors, width=.59, yerr=errors,
           error_kw={'elinewidth': .85, 'capsize': 3, 'capthick': .85}, zorder=2)
    for i, val in enumerate(values):
        ax.text(i, float(d.macro_f1_ci_high.iloc[i]) + .009, f'{val:.3f}', ha='center', fontsize=9)
    ax.set_xticks(np.arange(len(ids)), names)
    ax.set_title(title, pad=12, fontweight='bold')
    ax.set_xlabel(desc, fontsize=8.5, labelpad=12)
    score_axis(ax)
axes[1].set_ylabel('')
fig.subplots_adjust(bottom=.24, top=.87, left=.09, right=.99, wspace=.17)
finish(fig, 'fig03_ablation_panels', 'CNN output-only ablations (seed 42). Panel A compares the original prefix (up to 512 tokens), formatting removal, and model/provider keyword masking on the full test set. Panel B compares the original prefix with an exact 256-token prefix on a separately trained and evaluated long-answer subset; all five answers in each retained group satisfy the length rule. Comparisons are valid within panels. The higher scores in Panel B do not identify a performance gain from restricting length because the training and test populations differ from Panel A. Error bars are conditional 95% group-bootstrap intervals (500 resamples).')

def matrix(run):
    d = pd.read_csv(DATA / 'runs' / run / 'confusion_matrix.csv', index_col=0)
    if list(d.index) != list(d.columns):
        raise ValueError(f'Mismatched confusion labels: {run}')
    return d.div(d.sum(axis=1), axis=0)

def draw_matrix(ax, run, title, labels=True):
    mat = matrix(run)
    img = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    for row in range(5):
        for col in range(5):
            v = float(mat.iloc[row, col])
            ax.text(col, row, f'{100*v:.1f}', ha='center', va='center', fontsize=8.4,
                    color='white' if v > .56 else '#132337')
    ax.set_xticks(range(5), mat.columns, rotation=35, ha='right')
    ax.set_yticks(range(5), mat.index)
    ax.tick_params(axis='both', length=0)
    ax.set_xlabel('Predicted source')
    ax.set_ylabel('True source')
    ax.set_title(title, pad=10, fontsize=10.5, fontweight='bold')
    for spine in ax.spines.values(): spine.set_visible(False)
    return img

fig, ax = plt.subplots(figsize=(5.7, 4.7), constrained_layout=True)
im = draw_matrix(ax, 'main_cnn_output_only_s42', 'CNN output-only, seed 42')
cbar = fig.colorbar(im, ax=ax, fraction=.047, pad=.04)
cbar.set_label('Share within true source')
cbar.set_ticks(np.arange(0, 1.01, .2))
finish(fig, 'fig04_main_confusion', 'Row-normalized test confusion matrix for the CNN output-only classifier with seed 42. Numbers are percentages; colors show row proportions on a common 0–1 scale. Each true-source row contains 1,727 answers. Diagonal entries are source-specific recall, not precision.')

fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.25), constrained_layout=True)
for ax, domain, cond, title in zip(axes.flat,
        ['coding', 'coding', 'factual_qa', 'factual_qa'],
        ['mixed', 'heldout', 'mixed', 'heldout'],
        ['Coding: matched mixed', 'Coding: target held out', 'Factual QA: matched mixed', 'Factual QA: target held out']):
    im = draw_matrix(ax, f'rq3_{domain}_{cond}_s42', title)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.030, pad=.025)
cbar.set_label('Share within true source')
cbar.set_ticks(np.arange(0, 1.01, .2))
finish(fig, 'fig05_cross_domain_confusions', 'Row-normalized confusion matrices for the four cross-domain CNN experiments (seed 42). Numbers are percentages and every panel uses the same color scale. Coding has 198 answers per true source; factual QA has 647. Within each domain, matched-mixed and target-held-out conditions share the identical target test set and equal training sample size. The matrices display which source labels account for transfer failures; they do not establish a linguistic mechanism.')

# 6. Optional resource comparison; timing values are observed on this CPU setup.
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), sharey=True)
for ax, mode, title in zip(axes, ['output_only', 'input_and_output'], ['Output only', 'Input + output']):
    for i, (arch, color, label) in enumerate([('cnn', BLUE, 'CNN'), ('bilstm', ORANGE, 'BiLSTM')]):
        d = metrics.loc[(metrics.rq == 'main') & (metrics.architecture == arch) & (metrics['mode'] == mode)].sort_values('seed')
        xx = d.seconds.to_numpy() / 3600
        ax.scatter(xx, d.macro_f1, c=color, s=36, label=label, edgecolors='white', linewidth=.6, zorder=3)
        ax.scatter([xx.mean()], [d.macro_f1.mean()], c=color, marker='D', s=58, edgecolors='black', linewidth=.7, zorder=4)
    ax.set_xscale('log')
    ax.set_xlim(.25, 35)
    ax.set_xticks([.5, 1, 5, 10, 25], ['0.5', '1', '5', '10', '25'])
    ax.minorticks_off()
    ax.set_title(title, pad=10, fontweight='bold')
    ax.set_xlabel('Recorded run time on CPU (hours; log scale)')
    score_axis(ax)
axes[1].set_ylabel('')
fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=BLUE, label='CNN seed'),Line2D([], [], marker='o', linestyle='', color=ORANGE, label='BiLSTM seed'),Line2D([], [], marker='D', linestyle='', markeredgecolor='black', color='#707A88', label='Seed mean')], loc='lower center', bbox_to_anchor=(.51, -.01), ncol=3, frameon=False)
fig.subplots_adjust(bottom=.26, top=.90, left=.09, right=.99, wspace=.15)
finish(fig, 'fig06_compute_optional', 'Observed CPU run time and test macro-F1 for the six fitted seeds within each input condition. Circles show individual seeds and diamonds show their means. Time uses the recorded seconds field in metrics.json and includes training plus the initial test evaluation, but excludes later bootstrap/report generation; it is a setup-specific measure, not a controlled architecture benchmark. The horizontal axis is logarithmic and the macro-F1 axis spans 0–1.')

(OUT / 'figure_captions.json').write_text(json.dumps(captions, indent=2), encoding='utf-8')
print(json.dumps({'figures': list(captions), 'directory': str(OUT)}, indent=2))
