"""Picture of the weighted agreement network S for the 118th House.
Left: network layout (force-directed on the full weights; only the strongest edges are drawn).
Right: S itself as a heatmap, members ordered by party then NOMINATE dimension 1.
Run from the repo root:  uv run python mixed_membership/plot_agreement_graph.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
sys.path.insert(0, str(Path(__file__).parent))
import common as c

CONGRESS, EDGE_QUANTILE = 118, 0.97          # draw the strongest 3% of pairs within each party
HERE = Path(__file__).parent
BLUE, RED, GREY = "#2166ac", "#b2182b", "#555555"

n_bills = len(pd.read_csv(c.ROOT / f"data/{CONGRESS}/bills.csv"))
V, mem = c.load_votes(CONGRESS, min_votes=int(np.ceil(n_bills / 2)))
S = c.agreement_matrix(V)
n = len(S)
party = mem.party_code.map({100: "D", 200: "R"}).fillna("I").values
col = np.where(party == "D", BLUE, np.where(party == "R", RED, GREY))
nom = mem.nominate_dim1.values.astype(float)

# --- layout: spring layout on the full weighted graph; raising S to a power sharpens within vs across party
W = S ** 8
G = nx.from_numpy_array(W)
pos = nx.spring_layout(G, weight="weight", seed=0, iterations=300, k=1.5 / np.sqrt(n))
xy = np.array([pos[i] for i in range(n)])
# rotate so the long axis is horizontal, Democrats on the left
u, sv, vt = np.linalg.svd(xy - xy.mean(0), full_matrices=False)
xy = (xy - xy.mean(0)) @ vt.T
if xy[party == "D", 0].mean() > xy[party == "R", 0].mean():
    xy[:, 0] *= -1

iu = np.triu_indices(n, 1)
same = party[iu[0]] == party[iu[1]]
mask = np.zeros(len(iu[0]), bool)
thr = {}
for g in ("D", "R"):                              # strongest edges within each party (Republicans agree more, so one
    sel = same & (party[iu[0]] == g)              # global cutoff would leave the Democratic block empty)
    thr[g] = np.quantile(S[iu][sel], EDGE_QUANTILE)
    mask |= sel & (S[iu] >= thr[g])
cross = ~same
print("max cross-party S:", round(float(S[iu][cross].max()), 3), "| thresholds D, R:", round(float(thr["D"]), 3), round(float(thr["R"]), 3))
ei, ej = iu[0][mask], iu[1][mask]
w = S[ei, ej]
segs = np.stack([xy[ei], xy[ej]], axis=1)
alpha = 0.06 + 0.30 * (w - w.min()) / (w.max() - w.min() + 1e-12)
lc = LineCollection(segs, colors=[(0.35, 0.35, 0.35, a) for a in alpha], linewidths=0.4, zorder=1)

fig, axs = plt.subplots(2, 1, figsize=(9, 12.5), gridspec_kw={"height_ratios": [1, 1.9]})
ax = axs[0]
ax.add_collection(lc)
ax.scatter(*xy.T, c=col, s=22, edgecolor="white", linewidth=0.4, zorder=2)
ax.set_aspect("equal"); ax.axis("off"); ax.autoscale()
ax.scatter([], [], c=BLUE, s=30, label="Democrat"); ax.scatter([], [], c=RED, s=30, label="Republican")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)
ax.set_title(f"Agreement network: {n} members\n"
             f"{int(mask.sum()):,} edges drawn = strongest 3% of pairs within each party (of {len(S[iu]):,} pairs)\n"
             "layout uses all weights; no cross-party pair is that strong", fontsize=9.5)

# --- heatmap of S ordered by party then NOMINATE dim 1
key = pd.DataFrame({"p": pd.Categorical(party, ["D", "I", "R"]), "nom": np.where(np.isnan(nom), 9, nom)})
order = key.sort_values(["p", "nom"]).index.values
M = S[np.ix_(order, order)].copy()
np.fill_diagonal(M, np.nan)
ax = axs[1]
im = ax.imshow(M, cmap="Greens", vmin=0.4, vmax=1.0, interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])
nD = int((party == "D").sum())
ax.axhline(nD - 0.5, color="k", lw=0.6); ax.axvline(nD - 0.5, color="k", lw=0.6)
ax.text(nD / 2, -8, "Democrats", ha="center", va="bottom", fontsize=9, color=BLUE)
ax.text(nD + (n - nD) / 2, -8, "Republicans", ha="center", va="bottom", fontsize=9, color=RED)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("share of shared votes cast the same way (S)")
ax.set_title("The weighted matrix S itself\n(members sorted by party, then NOMINATE dim 1)", fontsize=10, pad=24)

fig.suptitle(f"{CONGRESS}th House: weighted vote-agreement network (final-passage votes)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(HERE / "agreement_network_118.png", dpi=140)
print("n =", n, "| edges drawn:", int(mask.sum()))

# --- standalone picture of the matrix
fig2, ax2 = plt.subplots(figsize=(7.5, 6.6))
im2 = ax2.imshow(M, cmap="Greens", vmin=0.4, vmax=1.0, interpolation="nearest")
ax2.set_xticks([]); ax2.set_yticks([])
ax2.axhline(nD - 0.5, color="k", lw=0.6); ax2.axvline(nD - 0.5, color="k", lw=0.6)
ax2.text(nD / 2, -6, "Democrats", ha="center", va="bottom", fontsize=9, color=BLUE)
ax2.text(nD + (n - nD) / 2, -6, "Republicans", ha="center", va="bottom", fontsize=9, color=RED)
cb2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
cb2.set_label("Agreement rate")
ax2.set_title(f"Agreement matrix, {CONGRESS}th House ({n} x {n})", pad=22)
fig2.tight_layout()
fig2.savefig(HERE / "agreement_matrix_118.png", dpi=140)
