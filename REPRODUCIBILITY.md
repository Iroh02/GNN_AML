# Reproducing the Results in the Paper

This document lists the exact commands that produce every number in
*"A Graph Neural Network for Explainable Anti-Money Laundering in Blockchain
Transactions"* (AIxB 2026, submission 18), together with expected runtimes and
the result files each command writes.

Every reported metric is the **mean ± sample standard deviation over five
seeds** `{0, 1, 7, 42, 123}`. Significance tests are **paired** across those
seeds (see [Statistical protocol](#statistical-protocol)).

---

## 1. Environment

```bash
pip install -r requirements.txt
```

Reference environment used for the reported numbers:

| Component | Version |
|---|---|
| OS | Windows 11 (26200) |
| Python | 3.14.2 |
| PyTorch | 2.10.0+cu128 |
| PyTorch Geometric | 2.6+ |
| scikit-learn | 1.8.0 |
| SciPy | 1.17.0 |
| GPU | NVIDIA RTX 5060 Laptop (8 GB VRAM) |

A CUDA GPU is not required — every script falls back to CPU — but the HBTBD
runs are roughly 10× slower on CPU.

> **Note on exact reproducibility.** Seeds are set for `torch`, `torch.cuda`
> and `numpy` immediately before model construction, which makes runs
> repeatable on *the same* hardware and library versions. cuDNN kernel
> selection is not pinned to deterministic algorithms, so metrics on different
> GPUs or PyTorch builds may differ in the third decimal place. This does not
> affect any conclusion in the paper, all of which are stated with explicit
> confidence intervals.

---

## 2. Datasets

Neither dataset is redistributed in this repository.

**Elliptic** — download `elliptic_txs_features.csv`,
`elliptic_txs_edgelist.csv` and `elliptic_txs_classes.csv` from the
Elliptic Data Set on Kaggle and place them in `data/raw/`.

**HBTBD** — obtain from Song & Gu (2023), *Applied Sciences* 13(15):8766,
doi:[10.3390/app13158766](https://doi.org/10.3390/app13158766), and unpack so
that the official split lives at:

```
data/hbtbd/HBTBD/data/train/
data/hbtbd/HBTBD/data/test/
```

Expected sizes: Elliptic raw CSVs ≈ 690 MB, HBTBD ≈ 188 MB.

---

## 3. Commands, in the order the paper uses them

### Table I & II — Elliptic homogeneous models and the timestep ablation

```bash
python run_multiseed_elliptic.py
```

Runs Logistic Regression, Static GraphSAGE, and the 2×2 cross of
{timestep encoding} × {skip connections}, five seeds each (25 runs).

- **Writes:** `results/multiseed_elliptic.json`, `results/multiseed_elliptic.csv`
- **Runtime:** ≈ 25 min on the reference GPU. The first invocation parses the
  690 MB feature CSV and caches tensors to
  `data/processed/elliptic_multiseed_cache.pt`; later runs skip that step.

### HBTBD main table — models, ensembles, tuned thresholds (n=10)

```bash
python run_multiseed_extended.py
```

Trains SimpleHeteroGNN, RiskMAGNN (d=128, L=2) and RiskMAGNN (d=192, L=3)
under one identical protocol, **ten** seeds each (30 runs): the original five
`{0, 1, 7, 42, 123}` plus five fixed before any extended result was observed
`{11, 21, 77, 2024, 31337}`. Saves per-run validation/test probabilities,
then computes per-architecture 10-seed probability ensembles and
validation-tuned F1 thresholds (a reported negative result — tuned thresholds
transfer worse than the default under the val–test shift).

- **Writes:** `results/multiseed_extended.json/.csv`,
  `results/ensemble_results.json`, `results/probs/*.npz`
- **Runtime:** ≈ 20 min

(`run_multiseed_all.py` is the original 5-seed version of the same protocol;
its results are retained in `results/multiseed_all.json` and are the first
five seeds of the extended run.)

### Feature-only baselines (the deconfounding controls)

```bash
python run_mlp_baseline.py   # Elliptic, 5 seeds
python run_mlp_hbtbd.py      # HBTBD, 10 seeds
```

Deep MLPs matched to the GNNs in depth, width, loss, optimiser, schedule and
seeds, but given **no graph input**. These separate "model class" from
"message passing" in the headline comparison: on Elliptic the MLP recovers
+37.11 of the 37.56 pp attributed to graph structure; on HBTBD it is
statistically indistinguishable from the heterogeneous GNNs on PR-AUC.

- **Writes:** `results/mlp_baseline_elliptic.json/.csv`,
  `results/mlp_baseline_hbtbd.json/.csv`
- **Runtime:** ≈ 3 min and ≈ 6 min

### Component ablation (n=10)

```bash
python run_component_ablation.py
```

2×2 design over `use_transe` × `use_risk_bias` on the RiskMAGNN d=128
skeleton, ten seeds each (40 runs). All four variants share residual
connections, LayerNorm, classifier head and training schedule, so the only
difference is which component is enabled.

- **Writes:** `results/component_ablation.json`, `results/component_ablation.csv`
- **Runtime:** ≈ 25 min

### All significance tests

```bash
python analyze_significance.py
```

Consumes the three JSON files above and emits every reported paired test.

- **Writes:** `results/significance_tests.json`, `results/significance_tests.txt`
- **Runtime:** < 5 s

### M2 metapath resplit diagnostic (n=5)

```bash
python resplit_hbtbd.py          # builds data/hbtbd_resplit/
python run_resplit_seeds.py      # RiskMAGNN Large, 5 seeds on the resplit
```

- **Writes:** `results/resplit_seeds.json/.csv`
- **Runtime:** ≈ 15 min
- **Note:** the resplit deliberately breaks temporal ordering and is *not* a
  performance claim; see the caveat in the paper.
  (`run_riskmagnn_resplit.py` is the original single-seed version.)

### Explanations

```bash
python explain_riskmagnn.py
```

- **Writes:** `explanations/`
- **Runtime:** ≈ 200 optimisation steps per transaction

### Learned attention weights (the 0.29 / 0.17 M2 numbers)

```bash
python check_attention.py
```

Loads the saved official-split and resplit checkpoints
(`models/riskmagnn_large.pth`, `models/riskmagnn_resplit.pth`) and prints the
mean per-metapath attention over layers and test nodes, plus the raw
`risk_bias` parameters. Requires the checkpoints and both datasets on disk.
Runtime: < 2 min on CPU.

> The submitted manuscript quoted a learned M2 attention weight of 0.184 for
> both the official split and the resplit; direct measurement of the
> checkpoints gives 0.29 (official) and 0.17 (resplit), and the camera-ready
> reports these measured values.

---

## 4. Statistical protocol

Implemented in `analyze_significance.py`.

- Each model is trained on the **same** five seeds over the **same** data, so
  seed *i* is a matched observation across models. Tests are therefore
  **paired two-sided t-tests**, which remove the seed-to-seed variance common
  to all models and are strictly more powerful than unpaired tests here.
- Reported alongside every comparison: mean difference, 95% confidence
  interval (t-based, n−1 df), p-value, and Cohen's *d_z*.
- **Wilcoxon signed-rank is computed but not used for inference.** With n = 5
  matched pairs its minimum attainable two-sided p-value is 2/2⁵ = 0.0625, so
  it can never reach p < 0.05 at this sample size. It is present in the JSON
  for completeness only; reading it as a null result would be a mistake.
- Primary metric is **PR-AUC** (average precision), appropriate under the
  heavy class imbalance in both datasets. ROC-AUC and F1 are reported
  alongside.

---

## 5. Corrections relative to the submitted version

The camera-ready supersedes several single-seed numbers from the submitted
manuscript. Both corrections are reproducible with the commands above.

| Submitted | Camera-ready (n = 10) | Why |
|---|---|---|
| RiskMAGNN 69.02% PR-AUC on HBTBD | 66.83 ± 4.04% (d=192); 10-seed ensemble 71.23% | The 69.02% came from one seed in a configuration where Base and Large were trained sequentially in a single process under one global seed. Under isolated re-seeding it does not reproduce; the multi-seed scripts re-seed before every model construction. |
| SimpleHeteroGNN 60.4% PR-AUC | 64.80 ± 1.22% | The original baseline used a weaker protocol (random validation split, identity metapath adjacencies) and so was never comparable to RiskMAGNN. It is retrained here under the unified protocol. |
| "+8.6 pp over SimpleHeteroGNN" | +2.03 pp, 95% CI [−0.47, +4.54], p = 0.099 | Not significant on PR-AUC. The significant gain is on ROC-AUC, for both sizes: +2.33 pp (d=128) and +2.53 pp (d=192), p = 0.002 each. |
| "Removing temporal encoding improves PR-AUC by 6.7 pp" | Not significant; sign flips with architecture | With skip connections: −1.63 pp (p = 0.23). Without: +4.47 pp (p = 0.37). The claim is withdrawn. |
| Component ablation (added for camera-ready) | Joint effect +2.72 pp, p = 0.057 at n=10 | On the first five seeds the effect was +3.84 pp (p = 0.020); one collapse seed in the extension moved it to marginal. Reported as suggestive, not confirmed. |

`results/riskmagnn_results.csv` and `results/hbtbd_results.csv` are retained
in the repository as the historical single-seed artefacts they are; the files
backing the camera-ready are `multiseed_extended.json`,
`multiseed_elliptic.json`, `component_ablation.json`, `resplit_seeds.json`,
`ensemble_results.json` and `significance_tests.json`.

---

## 6. Known discrepancy to be aware of

The submitted manuscript described the forensic attention priors as
π = [0.6, 0.6, 0.4] for (M1, M2, M3). The implementation in
`src/models/riskmagnn.py` initialises them at **[0.3, 0.6, 0.1]**
(M2 > M1 > M3). All reported results come from the implementation, and the
camera-ready text has been corrected to match the code.
