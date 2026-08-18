"""
Statistical Significance Analysis of the Multi-Seed Runs
=======================================================

Consumes the two multi-seed result files and reports, for every pairwise
model/variant comparison:

  * mean +/- sample std per model
  * paired mean difference with a 95% CI (t-based, n-1 df)
  * paired two-sided t-test p-value
  * Wilcoxon signed-rank p-value (reported with the caveat below)
  * Cohen's d_z (paired effect size)

Why PAIRED tests: every model is trained on the same 5 seeds over the same
data, so seed i is a matched observation across models. Pairing removes the
seed-to-seed variance that is common to all models and is strictly more
powerful than an unpaired Welch test here.

Caveat on Wilcoxon: with n=5 matched pairs the smallest attainable two-sided
p-value is 2/2^5 = 0.0625, so a Wilcoxon test can NEVER reach p<0.05 at this
sample size. It is reported for completeness only; the paired t-test and the
CI are the informative statistics. This is stated explicitly so the number is
not misread as a null result.

Inputs : results/multiseed_all.json, results/component_ablation.json
Outputs: results/significance_tests.json, results/significance_tests.txt
"""

import os
import sys
import json
import itertools

import numpy as np
from scipy import stats

ALPHA = 0.05
METRICS = [
    ('test_pr_auc', 'PR-AUC'),
    ('test_roc_auc', 'ROC-AUC'),
    ('test_f1', 'F1'),
]


def load(path: str, key: str):
    """Load a multi-seed result file into {group: {seed: {metric: value}}}."""
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        blob = json.load(f)
    table = {}
    for rec in blob['per_run']:
        if 'test_f1_argmax' in rec and 'test_f1' not in rec:
            rec = {**rec, 'test_f1': rec['test_f1_argmax']}
        table.setdefault(rec[key], {})[rec['seed']] = rec
    return blob, table


def paired_compare(a_vals, b_vals):
    """Paired stats for a - b. Inputs are seed-aligned arrays."""
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    d = a - b
    n = len(d)
    mean_d = float(d.mean())
    sd_d = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd_d / np.sqrt(n) if sd_d > 0 else 0.0

    if sd_d == 0:
        t_stat, p_t = (np.inf if mean_d else 0.0), (0.0 if mean_d else 1.0)
        lo = hi = mean_d
    else:
        t_stat, p_t = stats.ttest_rel(a, b)
        crit = stats.t.ppf(1 - ALPHA / 2, n - 1)
        lo, hi = mean_d - crit * se, mean_d + crit * se

    # Wilcoxon needs at least one non-zero difference.
    try:
        _, p_w = stats.wilcoxon(a, b)
    except ValueError:
        p_w = float('nan')

    d_z = mean_d / sd_d if sd_d > 0 else float('nan')

    return {
        'n_seeds': n,
        'mean_diff': mean_d,
        'std_diff': sd_d,
        'ci95_low': float(lo),
        'ci95_high': float(hi),
        't_stat': float(t_stat),
        'p_paired_t': float(p_t),
        'p_wilcoxon': float(p_w),
        'cohens_dz': float(d_z),
        'significant_at_05': bool(p_t < ALPHA),
    }


def analyse(table, group_key, title, out_lines, results):
    groups = list(table.keys())
    seeds = sorted(set(table[groups[0]]).intersection(*[set(table[g]) for g in groups]))

    out_lines.append("=" * 78)
    out_lines.append(f"  {title}")
    out_lines.append(f"  Matched seeds: {seeds}  (n={len(seeds)})")
    out_lines.append("=" * 78)

    section = {'seeds': seeds, 'descriptive': {}, 'comparisons': {}}

    # Descriptive statistics
    out_lines.append("")
    out_lines.append("  Descriptive statistics (mean +/- sample std, %)")
    out_lines.append("  " + "-" * 74)
    out_lines.append("  {:<28}{:>15}{:>15}{:>15}".format(
        "Model", "PR-AUC", "ROC-AUC", "F1"))
    for g in groups:
        cells = []
        section['descriptive'][g] = {}
        for mkey, _ in METRICS:
            v = np.array([table[g][s][mkey] for s in seeds]) * 100
            section['descriptive'][g][mkey] = {
                'mean': float(v.mean()), 'std': float(v.std(ddof=1))}
            cells.append(f"{v.mean():.2f}+/-{v.std(ddof=1):.2f}")
        out_lines.append("  {:<28}{:>15}{:>15}{:>15}".format(g, *cells))

    # Pairwise paired tests
    for mkey, mlabel in METRICS:
        out_lines.append("")
        out_lines.append(f"  Paired comparisons on {mlabel} (percentage points)")
        out_lines.append("  " + "-" * 74)
        out_lines.append("  {:<40}{:>10}{:>18}{:>9}".format(
            "Comparison (A - B)", "diff", "95% CI", "p(t)"))
        for a, b in itertools.combinations(groups, 2):
            av = [table[a][s][mkey] * 100 for s in seeds]
            bv = [table[b][s][mkey] * 100 for s in seeds]
            res = paired_compare(av, bv)
            section['comparisons'].setdefault(mkey, {})[f"{a} vs {b}"] = res
            star = " *" if res['significant_at_05'] else ""
            out_lines.append("  {:<40}{:>+10.2f}{:>18}{:>9.4f}{}".format(
                f"{a} - {b}",
                res['mean_diff'],
                f"[{res['ci95_low']:+.2f}, {res['ci95_high']:+.2f}]",
                res['p_paired_t'],
                star,
            ))
        out_lines.append("  (* = significant at alpha=0.05, paired two-sided t-test)")

    out_lines.append("")
    results[title] = section


def main():
    out_lines = []
    results = {}

    out_lines.append("STATISTICAL SIGNIFICANCE ANALYSIS -- RiskMAGNN on HBTBD")
    out_lines.append("")
    out_lines.append("All tests are PAIRED across seeds: each model is trained on the same")
    out_lines.append("5 seeds over identical data, so seed i is a matched observation.")
    out_lines.append("")
    out_lines.append("NOTE on Wilcoxon signed-rank: with n=5 matched pairs the minimum")
    out_lines.append("attainable two-sided p-value is 2/2^5 = 0.0625. A Wilcoxon test can")
    out_lines.append("therefore never reach p<0.05 here; it is reported in the JSON for")
    out_lines.append("completeness but the paired t-test and the 95% CI are the informative")
    out_lines.append("statistics at this sample size.")
    out_lines.append("")

    blob0, table0 = load('results/multiseed_extended.json', 'model')
    if table0 is not None:
        analyse(table0, 'model',
                'Model comparison, n=10 (official HBTBD split, extended seeds)',
                out_lines, results)

    blob, table = load('results/multiseed_all.json', 'model')
    if table is None:
        print("Missing results/multiseed_all.json -- run run_multiseed_all.py first.")
        return 1
    analyse(table, 'model',
            'Model comparison, n=5 (official HBTBD split, original run)',
            out_lines, results)

    blob2, table2 = load('results/component_ablation.json', 'variant')
    if table2 is None:
        out_lines.append("=" * 78)
        out_lines.append("  Component ablation not found "
                         "(run run_component_ablation.py).")
        out_lines.append("=" * 78)
    else:
        analyse(table2, 'variant',
                'Component ablation (RiskMAGNN Base skeleton, 2x2 design)',
                out_lines, results)

    blob3, table3 = load('results/multiseed_elliptic.json', 'variant')
    if table3 is None:
        out_lines.append("=" * 78)
        out_lines.append("  Elliptic multi-seed not found "
                         "(run run_multiseed_elliptic.py).")
        out_lines.append("=" * 78)
    else:
        analyse(table3, 'variant',
                'Elliptic chronological split (homogeneous models + '
                'temporal ablation)',
                out_lines, results)

    text = "\n".join(out_lines)
    print(text)

    os.makedirs('results', exist_ok=True)
    with open('results/significance_tests.txt', 'w') as f:
        f.write(text + "\n")
    with open('results/significance_tests.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved: results/significance_tests.txt and results/significance_tests.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
