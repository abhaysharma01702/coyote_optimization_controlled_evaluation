"""Recompute every Wilcoxon p-value in the paper from the raw paired scores.

Exact two-sided signed-rank p by full enumeration of sign flips (independent of SciPy), plus zero/tie diagnostics
and SciPy's own result for comparison (Decision/019).
Run from the repository root:  ccoa/Scripts/python.exe Code/Newest/tools/paper_figures/verify_wilcoxon.py"""
import itertools, json, sys, zipfile
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon, rankdata

sys.stdout.reconfigure(encoding="utf-8")
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RES = os.path.join(ROOT, "Code", "Newest", "Results")
AGG = os.path.join(ROOT, "Paper", "aggregate", "tables", "optimizer_all_runs.csv")


def exact_p(d, zero="wilcox"):
    d = np.asarray(d, float)
    if zero == "wilcox":
        d = d[d != 0]
    n = len(d)
    r = rankdata(np.abs(d))            # average ranks for ties
    wplus = r[d > 0].sum()
    tot = r.sum(); obs = min(wplus, tot - wplus)
    cnt = 0
    for signs in itertools.product((0, 1), repeat=n):
        w = r[np.array(signs, bool)].sum()
        if min(w, tot - w) <= obs + 1e-9:
            cnt += 1
    return cnt / 2 ** n, n


def report(name, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    zeros = int((d == 0).sum())
    ties = len(np.abs(d[d != 0])) - len(np.unique(np.round(np.abs(d[d != 0]), 12)))
    p_ex, n_used = exact_p(d)
    sp = wilcoxon(a, b)
    print(f"{name:28s} n={len(d)} zeros={zeros} tied|d|={ties} exact-enum p={p_ex:.4f} (n used {n_used})  scipy p={sp.pvalue:.4f}")


print("scipy", scipy.__version__)
print("== Ten-seed matched comparison (NB06a-c, NB09) ==")
for z, ds in (("6a.zip", "Amazon"), ("6b.zip", "Yelp"), ("6c.zip", "Twitter")):
    zf = zipfile.ZipFile(f"{RES}/{z}")
    for m in ("ann", "cnn"):
        d = json.loads(zf.read([n for n in zf.namelist() if n.endswith(f"{m}_wilcoxon.json")][0]))
        report(f"{ds} {m}", d["coa_f1"], d["def_f1"])
zf = zipfile.ZipFile(f"{RES}/09.zip")
ff = json.loads(zf.read([n for n in zf.namelist() if n.endswith("wilcoxon_result.json")][0]))
for m in ("ann", "cnn"):
    report(f"Fine Food {m}", ff[m]["coa_f1"], ff[m]["def_f1"])

print("== Optimizer comparison, 12 blocks (raw p before Holm) ==")
df = pd.read_csv(AGG)
for col, label in (("best_val_f1", "val"), ("F1-score", "test")):
    piv = df[df.optimizer != "DEFAULT"].pivot_table(index=["dataset", "seed"], columns="optimizer", values=col)
    for o in ("PSO", "RS", "TPE"):
        report(f"{label}: COA vs {o}", piv["COA"], piv[o])


# Ten-run comparison, two-sample view (Decision/019): the two arms of a "pair" use different seeds, so the runs are
# independent groups. Exact two-sided Mann-Whitney U, and Holm correction over the 8 comparisons for both tests.
from scipy.stats import mannwhitneyu


def holm(ps):
    order = np.argsort(ps); out = np.empty(len(ps)); run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, ps[k] * (len(ps) - i))); out[k] = run
    return out


print("== Ten-run comparison: signed-rank vs rank-sum, Holm over 8 ==")
names, sr, rs = [], [], []
for z, ds in (("6a.zip", "Amazon"), ("6b.zip", "Yelp"), ("6c.zip", "Twitter"), ("09.zip", "Fine Food")):
    zf = zipfile.ZipFile(f"{RES}/{z}")
    for m in ("ann", "cnn"):
        if ds == "Fine Food":
            d = json.loads(zf.read([n for n in zf.namelist() if n.endswith("wilcoxon_result.json")][0]))[m]
        else:
            d = json.loads(zf.read([n for n in zf.namelist() if n.endswith(f"{m}_wilcoxon.json")][0]))
        a, b = np.array(d["coa_f1"]), np.array(d["def_f1"])
        names.append(f"{ds} {m}"); sr.append(wilcoxon(a, b).pvalue)
        rs.append(mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)
for n, p1, h1, p2, h2 in zip(names, sr, holm(np.array(sr)), rs, holm(np.array(rs))):
    print(f"{n:16s} signed-rank p={p1:.4f} Holm={h1:.4f} | rank-sum p={p2:.4f} Holm={h2:.4f}")
