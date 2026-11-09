"""Scree plot for the 118th House: absolute eigenvalues of the weighted agreement matrix S
(linear scale, and log scale). Filled dots are positive eigenvalues, hollow dots are negative ones.
Run from the repo root:  uv run python mixed_membership/plot_scree_118.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import common as c

CONGRESS = 118
HERE = Path(__file__).parent
INK, MUTED, LINE = "#2b2b2b", "#777777", "#2166ac"

n_bills = len(pd.read_csv(c.ROOT / f"data/{CONGRESS}/bills.csv"))
V, mem = c.load_votes(CONGRESS, min_votes=int(np.ceil(n_bills / 2)))
S = c.agreement_matrix(V)
lam = np.linalg.eigvalsh(S)
lam = lam[np.argsort(-np.abs(lam))]            # ordered by absolute value, as in the lecture
a = np.abs(lam)
print("top 8 eigenvalues (signed):", np.round(lam[:8], 2))
print("log10 |lambda| for k=1..8:", np.round(np.log10(a[:8]), 2))

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, kmax, log in ((axs[0], 20, False), (axs[1], 40, True)):
    k = np.arange(1, kmax + 1)
    ax.plot(k, a[:kmax], "-", color=LINE, lw=1.2, zorder=1)
    pos = lam[:kmax] > 0
    ax.scatter(k[pos], a[:kmax][pos], s=34, color=LINE, zorder=2)
    ax.scatter(k[~pos], a[:kmax][~pos], s=34, facecolor="white", edgecolor=LINE, linewidth=1.3, zorder=2)
    if log:
        ax.set_yscale("log")
        ax.set_ylabel("|eigenvalue| (log scale)")
        ax.set_title(f"Log scale, first {kmax} eigenvalues", fontsize=11, color=INK)
    else:
        for kk in (1, 2, 3, 4):
            ax.annotate(f"{a[kk-1]:.1f}", (kk, a[kk - 1]), textcoords="offset points", xytext=(8, 4), fontsize=8, color=INK)
        ax.set_ylabel("|eigenvalue|")
        ax.set_title(f"Linear scale, first {kmax} eigenvalues", fontsize=11, color=INK)
    ax.set_xlabel("k (eigenvalue rank, by absolute value)")
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.suptitle(f"Scree plot of the agreement matrix S, {CONGRESS}th House (n = {len(S)})", fontsize=12, color=INK)
fig.text(0.5, 0.005, "Filled dot = positive eigenvalue, hollow dot = negative eigenvalue.",
         ha="center", fontsize=8.5, color=MUTED)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(HERE / "scree_118.png", dpi=140)

# --- scree for the speech covariates (428 x 5 matrix of speech-embedding PCs)
Xdf = c.speech_covariates(CONGRESS, mem.bioguide_id.values)
X = Xdf.fillna(Xdf.mean()).values
Xc = X - X.mean(0)
fig2, axs2 = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, (title, M) in zip(axs2, (("Raw (centered) PCs", Xc), ("Standardized PCs (what the model uses)", Xc / X.std(0)))):
    sv = np.linalg.svd(M, compute_uv=False)
    share = sv ** 2 / (sv ** 2).sum()
    k = np.arange(1, 6)
    ax.plot(k, share, "-o", color=LINE, lw=1.2, ms=6)
    for kk, sh in zip(k, share):
        ax.annotate(f"{sh*100:.0f}%", (kk, sh), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=INK)
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_xlabel("k (singular value rank)")
    ax.set_xticks(k); ax.set_ylim(0, 0.75)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs2[0].set_ylabel("share of total variance")
fig2.suptitle(f"Scree plot of the speech covariates, {CONGRESS}th House (n = {len(X)}, 5 PCs)", fontsize=12, color=INK)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(HERE / "scree_speech_118.png", dpi=140)
