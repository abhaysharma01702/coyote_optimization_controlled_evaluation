"""Two figures for Paper/main.tex that no notebook produces (Decision/019).

Reads only saved Kaggle outputs:
  Code/Newest/Results/6a.zip, 6b.zip, 6c.zip  -> test F1 of the ten runs per configuration (NB06a-c)
  Code/Newest/Results/09.zip                  -> the same for Amazon Fine Food (NB09)
  Paper/aggregate/tables/optimizer_all_runs.csv -> every optimizer-comparison final model (NB12)
Writes:
  Paper/fig_ten_runs.pdf         the ten COA and ten default runs of all 8 comparisons (no pairing)
  Paper/fig_selected_configs.pdf learning rate vs test F1 of the configuration each search selected

Run from the repository root:  ccoa/Scripts/python.exe Code/Newest/tools/paper_figures/make_paper_figures.py
"""
import json
import os
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
RES = os.path.join(ROOT, "Code", "Newest", "Results")
PAPER = os.path.join(ROOT, "Paper")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


def read_json(zip_name, suffix):
    zf = zipfile.ZipFile(os.path.join(RES, zip_name))
    name = [n for n in zf.namelist() if n.endswith(suffix)][0]
    return json.loads(zf.read(name))


def ten_runs():
    """Both groups of runs per comparison, centred on the default's mean. The runs of the two configurations use
    different seeds (independent groups), so no per-run differences are plotted."""
    groups = []
    for z, ds in (("6a.zip", "Amazon"), ("6b.zip", "Yelp"), ("6c.zip", "Twitter")):
        for model in ("ann", "cnn"):
            d = read_json(z, f"{model}_wilcoxon.json")
            groups.append((np.array(d["coa_f1"]), np.array(d["def_f1"])))
    ff = read_json("09.zip", "wilcoxon_result.json")
    for model in ("ann", "cnn"):
        groups.append((np.array(ff[model]["coa_f1"]), np.array(ff[model]["def_f1"])))
    for coa, dfl in groups:
        assert len(coa) == len(dfl) == 10

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    rng = np.random.RandomState(0)
    for i, (coa, dfl) in enumerate(groups):
        ref = dfl.mean()
        for off, vals, c, lab in ((-0.18, dfl, "#999999", "default"), (0.18, coa, "#1f77b4", "COA")):
            ax.scatter(i + off + rng.uniform(-0.07, 0.07, 10), vals - ref, s=8, color=c, lw=0, alpha=0.85,
                       label=lab if i == 0 else None)
            ax.plot([i + off - 0.13, i + off + 0.13], [vals.mean() - ref] * 2, color="black", lw=1.2)
    for k in range(1, 4):
        ax.axvline(2 * k - 0.5, color="lightgrey", lw=0.6)
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["ANN", "CNN"] * 4, fontsize=6.5)
    for k, ds in enumerate(("Amazon", "Yelp", "Twitter", "Fine Food")):
        ax.text(2 * k + 0.5, -0.13, ds, ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(-0.022, 0.022)
    ax.set_ylabel("Test F1 $-$ mean of default runs")
    ax.legend(loc="upper left", ncol=2, frameon=False, fontsize=6.5, handletextpad=0.2, columnspacing=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    out = os.path.join(PAPER, "fig_ten_runs.pdf")
    fig.savefig(out)
    print("wrote", out, [round(float(c.mean() - d.mean()), 4) for c, d in groups])


def selected_configs():
    df = pd.read_csv(os.path.join(PAPER, "aggregate", "tables", "optimizer_all_runs.csv"))
    assert len(df) == 60, len(df)          # 3 datasets x 4 seeds x (4 optimizers + default)
    style = {"COA": ("o", "#1f77b4"), "PSO": ("s", "#ff7f0e"), "RS": ("^", "#2ca02c"), "TPE": ("D", "#d62728")}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    for ax, ds in zip(axes, ("amazon", "yelp", "twitter")):
        sub = df[df["dataset"] == ds]
        dfl = sub[sub["optimizer"] == "DEFAULT"]["F1-score"]
        m, s = dfl.mean(), dfl.std()
        ax.axhspan(m - s, m + s, color="grey", alpha=0.18, lw=0)
        ax.axhline(m, color="grey", lw=1, label="default (mean $\\pm$ sd)")
        for opt, (mk, c) in style.items():
            o = sub[sub["optimizer"] == opt]
            ax.scatter(o["lr"], o["F1-score"], marker=mk, s=16, color=c, label=opt, alpha=0.85, lw=0)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1e-2)
        ax.set_title(ds.capitalize())
        ax.set_xlabel("Selected learning rate $\\eta$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Test F1 of final model")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = os.path.join(PAPER, "fig_selected_configs.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ten_runs()
    selected_configs()
