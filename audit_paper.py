"""Recompute every number quoted in paper_camera_ready.tex from the result files."""
import json
import numpy as np
from scipy import stats

R = r"C:\Users\nandi\Desktop\GNN_AML\results"


def load(name):
    with open(rf"{R}\{name}") as f:
        return json.load(f)


def col(per, key, field=None, value=None):
    rows = [r for r in per if (field is None or r.get(field) == value)]
    rows.sort(key=lambda r: r['seed'])
    return np.array([r[key] for r in rows]) * 100, [r['seed'] for r in rows]


def paired(a, b):
    d = a - b
    n = len(d)
    t, p = stats.ttest_rel(a, b)
    se = d.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    return d.mean(), d.mean() - crit * se, d.mean() + crit * se, p


def show(label, claim, got):
    ok = "OK " if abs(claim - got) < 0.06 else "MISMATCH"
    print(f"  [{ok}] {label:<52} paper={claim:>8.2f}  computed={got:>8.2f}")


print("=" * 78)
print("TABLE: ELLIPTIC (5 seeds)")
print("=" * 78)
ell = load("multiseed_elliptic.json")["per_run"]
mlp_e = load("mlp_baseline_elliptic.json")["per_run"]
for name, pr_c, roc_c in [
    ("LogisticRegression", 16.21, 82.44),
    ("StaticGraphSAGE", 53.77, 81.75),
    ("TemporalGNN", 52.14, 81.59),
    ("TemporalGNN (no skip)", 43.39, 81.62),
    ("No temporal (no skip)", 38.92, 79.24),
]:
    pr, _ = col(ell, "test_pr_auc", "variant", name)
    roc, _ = col(ell, "test_roc_auc", "variant", name)
    show(f"{name} PR", pr_c, pr.mean())
    show(f"{name} ROC", roc_c, roc.mean())
mpr = np.array([r["test_pr_auc"] for r in sorted(mlp_e, key=lambda r: r['seed'])]) * 100
mroc = np.array([r["test_roc_auc"] for r in sorted(mlp_e, key=lambda r: r['seed'])]) * 100
show("Deep MLP PR", 53.32, mpr.mean())
show("Deep MLP PR std", 3.67, mpr.std(ddof=1))
show("Deep MLP ROC", 81.87, mroc.mean())

print("\n  Claims in Elliptic text:")
lr, _ = col(ell, "test_pr_auc", "variant", "LogisticRegression")
sage, _ = col(ell, "test_pr_auc", "variant", "StaticGraphSAGE")
tg, _ = col(ell, "test_pr_auc", "variant", "TemporalGNN")
tgn, _ = col(ell, "test_pr_auc", "variant", "TemporalGNN (no skip)")
nn, _ = col(ell, "test_pr_auc", "variant", "No temporal (no skip)")
d, lo, hi, p = paired(sage, lr)
show("LogReg->GraphSAGE +37.56", 37.56, d)
show("  CI low +36.34", 36.34, lo)
show("  CI high +38.78", 38.78, hi)
d, lo, hi, p = paired(mpr, lr)
show("MLP recovers +37.11", 37.11, d)
d, lo, hi, p = paired(sage, mpr)
show("GraphSAGE - MLP +0.45", 0.45, d)
show("  CI low -5.04", -5.04, lo)
show("  CI high +5.94", 5.94, hi)
print(f"       p = {p:.4f} (paper says 0.83)")
d, lo, hi, p = paired(sage, tg)
show("timestep w/ skip: 1.63", 1.63, d)
print(f"       p = {p:.4f} (paper says 0.23)")
d, lo, hi, p = paired(tgn, nn)
show("timestep no-skip: 4.47", 4.47, d)
print(f"       p = {p:.4f} (paper says 0.37)")
d, lo, hi, p = paired(tg, tgn)
show("skip w/ timestep +8.75", 8.75, d)
show("  CI low +2.10", 2.10, lo)
show("  CI high +15.41", 15.41, hi)
print(f"       p = {p:.4f} (paper says 0.022)")
d, lo, hi, p = paired(sage, nn)
show("skip w/o timestep +14.85", 14.85, d)
print(f"       p = {p:.4f} (paper says 0.007)")

print("\n" + "=" * 78)
print("TABLE: HBTBD (10 seeds)")
print("=" * 78)
ext = load("multiseed_extended.json")["per_run"]
mlp_h = load("mlp_baseline_hbtbd.json")["per_run"]
ens = load("ensemble_results.json")
mh_pr = np.array([r["test_pr_auc"] for r in sorted(mlp_h, key=lambda r: r['seed'])]) * 100
mh_roc = np.array([r["test_roc_auc"] for r in sorted(mlp_h, key=lambda r: r['seed'])]) * 100
mh_f1 = np.array([r["test_f1"] for r in sorted(mlp_h, key=lambda r: r['seed'])]) * 100
show("MLP PR", 64.70, mh_pr.mean())
show("MLP PR std", 2.24, mh_pr.std(ddof=1))
show("MLP ROC", 88.65, mh_roc.mean())
show("MLP F1", 67.07, mh_f1.mean())
for name, pr_c, roc_c, f1_c in [
    ("SimpleHeteroGNN", 64.80, 87.62, 63.24),
    ("RiskMAGNN (Base)", 63.50, 89.95, 62.33),
    ("RiskMAGNN (Large)", 66.83, 90.15, 64.53),
]:
    pr, _ = col(ext, "test_pr_auc", "model", name)
    roc, _ = col(ext, "test_roc_auc", "model", name)
    f1, _ = col(ext, "test_f1_argmax", "model", name)
    show(f"{name} PR", pr_c, pr.mean())
    show(f"{name} ROC", roc_c, roc.mean())
    show(f"{name} F1", f1_c, f1.mean())
show("ens SimpleHetero PR", 70.18, ens["SimpleHeteroGNN"]["argmax"]["pr_auc"] * 100)
show("ens RiskMAGNN Large PR", 71.23, ens["RiskMAGNN (Large)"]["argmax"]["pr_auc"] * 100)
show("ens RiskMAGNN Large ROC", 91.17, ens["RiskMAGNN (Large)"]["argmax"]["roc_auc"] * 100)
show("ens RiskMAGNN Large F1", 68.12, ens["RiskMAGNN (Large)"]["argmax"]["f1"] * 100)

print("\n  Claims in HBTBD text:")
sh, _ = col(ext, "test_pr_auc", "model", "SimpleHeteroGNN")
rb, _ = col(ext, "test_pr_auc", "model", "RiskMAGNN (Base)")
rl, _ = col(ext, "test_pr_auc", "model", "RiskMAGNN (Large)")
sh_r, _ = col(ext, "test_roc_auc", "model", "SimpleHeteroGNN")
rb_r, _ = col(ext, "test_roc_auc", "model", "RiskMAGNN (Base)")
rl_r, _ = col(ext, "test_roc_auc", "model", "RiskMAGNN (Large)")
d, lo, hi, p = paired(sh, mh_pr)
show("SimpleHetero - MLP PR +0.09", 0.09, d)
print(f"       p = {p:.4f} (paper says 0.92)")
d, lo, hi, p = paired(rl, mh_pr)
show("RiskMAGNN L - MLP PR +2.13", 2.13, d)
print(f"       p = {p:.4f} (paper says 0.22)")
d, lo, hi, p = paired(sh_r, mh_roc)
show("SimpleHetero - MLP ROC -1.02", -1.02, d)
print(f"       p = {p:.4f} (paper says 0.034)")
d, lo, hi, p = paired(rb_r, mh_roc)
show("RiskMAGNN B - MLP ROC +1.30", 1.30, d)
print(f"       p = {p:.4f} (paper says 0.006)")
d, lo, hi, p = paired(rl_r, mh_roc)
show("RiskMAGNN L - MLP ROC +1.50", 1.50, d)
print(f"       p = {p:.4f} (paper says 0.033)")
d, lo, hi, p = paired(rb_r, sh_r)
show("RiskMAGNN B - SH ROC +2.33", 2.33, d)
print(f"       p = {p:.4f} (paper says 0.002)")
d, lo, hi, p = paired(rl_r, sh_r)
show("RiskMAGNN L - SH ROC +2.53", 2.53, d)
print(f"       p = {p:.4f} (paper says 0.002)")

print("\n" + "=" * 78)
print("TABLE: ABLATION (10 seeds)")
print("=" * 78)
ab = load("component_ablation.json")["per_run"]
for name, pr_c, roc_c, f1_c in [
    ("Full (TransE + RiskBias)", 63.50, 89.95, 62.33),
    ("-TransE", 61.54, 90.51, 61.77),
    ("-RiskBias", 62.61, 90.46, 61.94),
    ("-Both", 60.77, 89.75, 60.62),
]:
    pr, _ = col(ab, "test_pr_auc", "variant", name)
    roc, _ = col(ab, "test_roc_auc", "variant", name)
    f1, _ = col(ab, "test_f1", "variant", name)
    show(f"{name} PR", pr_c, pr.mean())
    show(f"{name} ROC", roc_c, roc.mean())
    show(f"{name} F1", f1_c, f1.mean())
full, _ = col(ab, "test_pr_auc", "variant", "Full (TransE + RiskBias)")
both, _ = col(ab, "test_pr_auc", "variant", "-Both")
notr, _ = col(ab, "test_pr_auc", "variant", "-TransE")
d, lo, hi, p = paired(full, both)
show("Full - Both +2.72", 2.72, d)
show("  CI low -0.10", -0.10, lo)
show("  CI high +5.55", 5.55, hi)
print(f"       p = {p:.4f} (paper says 0.057)")
nine = int((full - both > 0).sum())
print(f"       direction positive in {nine}/10 seeds (paper says nine of ten)")
d, lo, hi, p = paired(full, notr)
print(f"       Full - (-TransE): {d:+.2f}, p={p:.4f}  [gating: no detectable benefit]")
print("  collapse seeds (min PR per variant):")
for name in ["Full (TransE + RiskBias)", "-TransE", "-RiskBias", "-Both"]:
    pr, sds = col(ab, "test_pr_auc", "variant", name)
    print(f"       {name:<26} min {pr.min():.2f} (seed {sds[int(pr.argmin())]})")

print("\n" + "=" * 78)
print("TABLE: RESPLIT")
print("=" * 78)
rs = load("resplit_seeds.json")["per_run"]
rv = np.array([r["val_pr_auc"] for r in rs]) * 100
rt = np.array([r["test_pr_auc"] for r in rs]) * 100
show("resplit val 97.5", 97.5, rv.mean())
show("resplit test 94.3", 94.3, rt.mean())
show("resplit test std 0.35", 0.35, rt.std(ddof=1))
show("resplit gap 3.2", 3.2, (rv - rt).mean())
lv, _ = col(ext, "val_pr_auc", "model", "RiskMAGNN (Large)")
lt, _ = col(ext, "test_pr_auc", "model", "RiskMAGNN (Large)")
show("official val 96.5", 96.5, lv.mean())
show("official test 66.8", 66.8, lt.mean())
show("official gap 29.7", 29.7, (lv - lt).mean())
mv = np.array([r["val_pr_auc"] for r in mlp_h]) * 100
mt = np.array([r["test_pr_auc"] for r in mlp_h]) * 100
show("MLP gap 34.7", 34.7, (mv - mt).mean())
show("MLP gap std 2.2", 2.2, (mv - mt).std(ddof=1))
