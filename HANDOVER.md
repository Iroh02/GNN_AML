# HANDOVER — AIxB 2026 Submission 18 (RiskMAGNN)

**Paper:** "A Graph Neural Network for Explainable Anti-Money Laundering in
Blockchain Transactions" — accepted as a Short Paper (4 pages).
**Authors:** Nandita Menon, Anushree Ashok, Jillian Priscilla, Debashis Guha
(S P Jain School of Global Management, Dubai).
**Deadline:** camera-ready **and** author registration, **21 Aug 2026 (AoE)**.
**Repo:** <https://github.com/Iroh02/GNN_AML> (public).
**Camera-ready source:** `paper_camera_ready.tex` (IEEEtran, `conference`).

---

## 1. Status in one paragraph

The camera-ready is content-complete and every claim in it has been verified
against the raw data or a primary source. The submitted (June) version
contained several claims that did not survive verification; all are corrected
and the corrections are stated openly in the paper rather than quietly
patched. **The one blocking task left is length:** the file is ~5,000 words
and will not fit 4 pages. See §2.

---

## 2. Outstanding before the deadline

| # | Task | Owner | Est. | Notes |
|---|---|---|---|---|
| 1 | **Trim to 4 pages** | — | 15 min | Cut order: (a) "Explanation Findings" subsection, (b) `SimpleHeteroGNN (ens.)` row in Table III, (c) the "In hindsight…" sentence in Discussion, (d) Related Work → 2 paragraphs. Check whether AIxB counts references inside the 4 pages. |
| 2 | **Compile on Overleaf** | Nandita | 10 min | No LaTeX locally. Verify: no table overflows a column, page count, equations fit. Structure already validated (balanced envs, all cites/refs resolve). |
| 3 | **Confirm Prof. Guha's email** | Nandita | 2 min | Student emails corrected to `as24dxb…@spjain.org`. His is currently `debashis.guha@spjain.org` — unverified, and the student format turned out to differ from what was submitted. |
| 4 | **Confirm reference [6] venue** | Nandita | 5 min | Prof. Guha's link was `ijcnc.com`, but the publisher abstract page is `aircconline.com/abstract/**ijnsa**/v17n6/` and says **IJNSA** 17(5/6). Paper currently cites IJNSA. Both are AIRCC journals. |
| 5 | **Sign off AI disclosure** | All authors | 5 min | §"Disclosure of AI-Generated Content". Reads accurately as written; it is an authorship statement, so the authors must own it. |
| 6 | **Author registration** | Nandita/Guha | ? | ⚠️ **Highest-risk item.** At least one author must complete *full* registration by 21 Aug AoE or the paper is dropped from the proceedings and IEEE Xplore. Involves payment/university admin — do not leave to the last hours. |
| 7 | **Upload camera-ready** | Nandita | 10 min | IEEE CPS link in the acceptance email. |

---

## 3. Corrections vs. the submitted version

All were found by verification, not by opinion. Each is stated in the paper
and in `RESPONSE_TO_REVIEWERS.md`.

| # | Submitted claim | Verified reality | How found |
|---|---|---|---|
| 1 | RiskMAGNN 69.02% PR-AUC | 66.83 ± 4.04 (d=192, 10 seeds). 69.02 never recurs on any seed. | `run_riskmagnn.py:35` seeds **once** at module level, then trains Base and Large sequentially — Large inherited leftover RNG state. |
| 2 | "+8.6 pp over SimpleHeteroGNN" | +2.03 pp, 95% CI [−0.47, +4.54], p = 0.099 — n.s. | Old baseline used a weaker protocol (random val split, identity metapath adjacencies) → never comparable. Retrained identically. |
| 3 | "Temporal modelling hurts AML" (+6.7 pp) | Not significant; **sign flips** with architecture (−1.63 pp with skips, +4.47 without). | Original ablation changed timestep *and* skip connections together. Clean 2×2 run. |
| 4 | Components jointly +3.84 pp (p = 0.020) | +2.72 pp, p = 0.057 at n = 10 — suggestive, not confirmed. | Seed extension 5 → 10; one collapse seed. |
| 5 | "Graph structure +37.6 pp" | It is **model class**, not message passing: a capacity-matched feature-only MLP recovers +37.11 of those points; GraphSAGE − MLP = +0.45, p = 0.83. Same on HBTBD (MLP 64.70 vs SimpleHeteroGNN 64.80, p = 0.92). | Prof. Guha's query → `run_mlp_baseline.py`, `run_mlp_hbtbd.py`. Features already contain 1-hop/2-hop aggregates. |
| 6 | "MAGNN discards intermediate nodes"; "RiskMAGNN explicitly models the wallet address" | Both false. MAGNN's instance encoders **do** encode intermediates. Our `t` is the mean of neighbouring **transaction** embeddings — no address data enters the model at all. | Prof. Guha's query → code path audit + `idx0*.pickle` inspection. |
| 7 | "M2 entirely absent from training (0 instances), one-way flows only in ts 35–49" | The **adjacency list** is empty (0 edges) — but `idx01.pickle` holds **602,850** M2 training instances (382,342 with distinct endpoints), source timesteps **1–34**. M1's two views match exactly (3,305,618 = 3,305,618), so this is an **internal inconsistency in the release**. | `build_address_features.py` |
| 8 | Refs [6],[7],[8] | **Fabricated** — wrong authors and titles. Corrected: Ferretti/D'Angelo/Ghini (IEEE Access 13:50201–50215); Deprez/Vanderschueren/Baesens/Verdonck/Verbeke (arXiv:2405.19383); Lawal/Okolie/Obunadike (IJNSA 17(5/6)). | Crossref, arXiv, publisher pages. All 10 refs re-verified. |
| 9 | Forensic priors π = [0.6, 0.6, 0.4] | Code uses **[0.3, 0.6, 0.1]** (`src/models/riskmagnn.py:110`). Paper now matches the code. | Code read. |
| 10 | Learned M2 attention = 0.184 | Measured **0.29** (official) / **0.17** (resplit); priors alone imply 0.43. | `check_attention.py` on saved checkpoints. |
| 11 | Author emails | Students use `firstname.as24dxbNNN@spjain.org`. | User correction. |

> **Nothing needed retraining.** Every model consumed the released adjacency
> lists, so all tables faithfully record a real, reproducible condition. Items
> 6 and 7 changed *explanations*, not measurements.

---

## 4. Final results (all verified — `audit_paper.py` reports zero mismatches)

**Elliptic (5 seeds, chronological split, test PR-AUC / ROC-AUC)**

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic Regression (no graph) | 16.21 ± 0.00 | 82.44 ± 0.00 |
| **Deep MLP (no graph)** | **53.32 ± 3.67** | 81.87 ± 0.23 |
| Skip, no timestep (GraphSAGE) | 53.77 ± 0.98 | 81.75 ± 0.38 |
| Skip, + timestep | 52.14 ± 1.79 | 81.59 ± 0.98 |
| No skip, + timestep | 43.39 ± 4.64 | 81.62 ± 0.80 |
| No skip, no timestep | 38.92 ± 6.96 | 79.24 ± 1.49 |

- LogReg → GraphSAGE: +37.56, CI [+36.34, +38.78], p < 10⁻⁴
- MLP recovers +37.11 of it; **GraphSAGE − MLP = +0.45, p = 0.83**
- Skip connections: +8.75 (p = 0.022) with timestep, +14.85 (p = 0.007) without
- Timestep effect: −1.63 (p = 0.23) with skips, +4.47 (p = 0.37) without → **sign flips, n.s.**

**HBTBD (10 seeds, official split)**

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Feature-only MLP | 64.70 ± 2.24 | 88.65 ± 0.77 | **67.07 ± 1.15** |
| SimpleHeteroGNN | 64.80 ± 1.22 | 87.62 ± 1.20 | 63.24 ± 1.82 |
| RiskMAGNN (d=128) | 63.50 ± 6.78 | 89.95 ± 1.04 | 62.33 ± 3.72 |
| RiskMAGNN (d=192) | 66.83 ± 4.04 | 90.15 ± 1.38 | 64.53 ± 3.22 |
| SimpleHeteroGNN (10-seed ens.) | 70.18 | 89.57 | 67.48 |
| **RiskMAGNN d=192 (10-seed ens.)** | **71.23** | **91.17** | **68.12** |

- PR-AUC: no single-model difference significant (RiskMAGNN d=192 − MLP = +2.13, p = 0.22)
- ROC-AUC: RiskMAGNN − MLP = +1.30 (d=128, p = 0.006) / +1.50 (d=192, p = 0.033);
  RiskMAGNN − SimpleHeteroGNN = +2.33 / +2.53, p = 0.002 both
- **SimpleHeteroGNN ranks *worse* than the MLP** (−1.02 ROC, p = 0.034)
- Validation-tuned F1 thresholds transfer **worse** than default (59.3 vs 63.2) — reported as a negative result

**Component ablation (10 seeds, d=128 skeleton)**

| Variant | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Full (both) | 63.50 ± 6.78 | 89.95 ± 1.04 | 62.33 ± 3.72 |
| − Gating | 61.54 ± 10.18 | 90.51 ± 1.03 | 61.77 ± 4.22 |
| − Risk bias | 62.61 ± 3.20 | 90.46 ± 0.47 | 61.94 ± 2.79 |
| − Both | 60.77 ± 4.04 | 89.75 ± 0.53 | 60.62 ± 3.27 |

- Full − Both: +2.72, CI [−0.10, +5.55], **p = 0.057** (9/10 seeds positive)
- Full − (−Gating): +1.96, p = 0.64 → no detectable benefit
- **Collapse seeds:** Full → 45.76 (seed 2024), −Gating → 34.46 (seed 0), −Both → 52.98 (seed 2024)

**Gap diagnostic (RiskMAGNN d=192)**

| Split | Val PR | Test PR | Gap |
|---|---|---|---|
| Official (temporal) | 96.5 ± 1.5 | 66.8 ± 4.0 | 29.7 ± 4.0 |
| Stratified resplit | 97.5 ± 1.4 | 94.3 ± 0.4 | 3.2 ± 1.4 |

The MLP shows a **34.7 ± 2.2 pp** gap with **no metapaths at all** → the gap is
temporal feature shift, not the missing relation type.

---

## 5. Repository map

**Experiment scripts (produce the paper's numbers)**

| Script | Produces | Runtime |
|---|---|---|
| `run_multiseed_elliptic.py` | Elliptic table (25 runs, 5 seeds) | ~25 min |
| `run_mlp_baseline.py` | Elliptic MLP control | ~5 min |
| `run_multiseed_extended.py` | HBTBD table + ensembles + tuned thresholds (30 runs, 10 seeds) | ~20 min |
| `run_mlp_hbtbd.py` | HBTBD MLP control (10 seeds) | ~10 min |
| `run_component_ablation.py` | Ablation table (40 runs, 10 seeds) | ~25 min |
| `run_resplit_seeds.py` | Resplit diagnostic (5 seeds) | ~15 min |
| `analyze_significance.py` | All paired t-tests / CIs | < 5 s |
| `check_attention.py` | Learned M2 attention (0.29 / 0.17) | < 2 min |
| `build_address_features.py` | Dataset audit + address aggregates | ~2 min |
| `audit_paper.py` | **Re-verifies every number in the paper** | < 5 s |

**Key result files:** `results/multiseed_elliptic.json`,
`multiseed_extended.json`, `component_ablation.json`, `resplit_seeds.json`,
`ensemble_results.json`, `mlp_baseline_elliptic.json`,
`mlp_baseline_hbtbd.json`, `significance_tests.{txt,json}`.

**Superseded (kept as historical artefacts only):**
`results/riskmagnn_results.csv` (the 69.02%), `results/hbtbd_results.csv`
(the 60.4%). Do not cite.

**Docs:** `REPRODUCIBILITY.md` (command per table), `RESPONSE_TO_REVIEWERS.md`
(point-by-point), `README.md` (headline results + superseded-numbers warning).

---

## 6. Gotchas for whoever picks this up

1. **Seed once, train twice = silent corruption.** The original bug. Always
   re-seed immediately before model construction (`set_seed(seed)` then build).
2. **Training collapses happen** — ~1 seed in 10 drops 15–20 PR-AUC points with
   *training loss decreasing normally*. Only validation reveals it. Never trust
   a single run on this benchmark.
3. **The features already encode the graph.** Elliptic/HBTBD node features
   include 1-hop and 2-hop aggregates (indices 41–164), which is why
   "feature-only" baselines are not graph-blind. Any GNN-vs-baseline claim on
   these datasets must control for model class.
4. **The loader discards data.** `load_hbtbd_data()` reads only
   `features0.npy` + `m*.adjlist`. The release also has `idx0*.pickle`
   (instance triples with address ids) and `features1-3.npy` (8-dim address
   features). Unused by every reported model.
5. **Neighbour cap = 50** (`run_riskmagnn.py:58`). The graph the model sees is
   truncated; disclosed in the paper.
6. **`.gitignore` previously excluded `results/` and `*.csv`** — i.e. the very
   evidence. Fixed; keep it that way.
7. **Overleaf does not sync with the local repo.** Re-upload after every edit.

---

## 7. Future work (in priority order)

1. **Retrain with M2 restored** from `idx01.pickle` — measures what the empty
   adjacency list actually costs. Never done; the paper says so.
2. **Implement a real MAGNN instance encoder** using the address ids and
   features. `build_address_features.py` already emits the per-node aggregates
   `T_v`; because mean-pooling and `W` are linear, the encoder reduces to
   `W[h_v + T_v⊙r₁ ; T_v ; N_v + T_v⊙r₂]` — no need to materialise millions of
   instance embeddings. This is the cheapest path to a genuine architectural
   contribution.
3. **Evaluate real temporal architectures** (TGN, TGAT) — currently untested,
   so no claim about temporal graph learning is supported.
4. **Raw-feature benchmark** without engineered neighbourhood aggregates, to
   test whether the feature-only result generalises beyond these datasets.
5. Amortised explainer (200 optimisation steps/transaction is too slow for
   triage); validate forensic priors against labelled typologies.

---

## 8. The honest summary of what this paper now says

Graph structure is not what separates the models on these benchmarks — model
class is, because the engineered features already carry the neighbourhood.
What survives for architecture is a modest, reliable **ranking** gain that
appears only with our aggregation (+1.3 to +2.5 ROC-AUC, p ≤ 0.03), while a
naive heterogeneous GNN ranks *worse* than a plain MLP. The best detector is
a 10-seed ensemble (71.2% average precision). The most transferable findings
are about the benchmark itself: HBTBD ships an empty M2 training adjacency
list while its own instance files contain that relation, and the official
split carries a temporal distribution shift that accounts for most of the
validation–test gap. Seed variance on this benchmark is comparable to the
architectural differences people publish.

That is a more useful short paper than the one submitted in June, and every
number in it can be regenerated from the released code.
