"""118th House: mixed memberships (Mixed-SCORE) on the weighted agreement matrix S plus speech PCs, tau=1.
Writes simplex_K3_118.png (triangle) and simplex_K4_118.png (tetrahedron) next to this file.
Run from the repo root:  uv run python mixed_membership/run_118.py"""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import common as c
from mixed_score import mixed_score

CONGRESS, TAU0 = 118, 1.0
HERE = Path(__file__).parent

n_bills = len(pd.read_csv(c.ROOT / f"data/{CONGRESS}/bills.csv"))
V, mem = c.load_votes(CONGRESS, min_votes=int(np.ceil(n_bills / 2)))      # at least half of the votes
S = c.agreement_matrix(V)
Xdf = c.speech_covariates(CONGRESS, mem.bioguide_id.values)
X = Xdf.fillna(Xdf.mean()).values
party = mem.party_code.map({100: "D", 200: "R"}).fillna("I").values
nom = mem.nominate_dim1.values.astype(float)
col = np.where(party == "D", "#2166ac", np.where(party == "R", "#b2182b", "#555"))
rng = np.random.default_rng(0)
print(f"{CONGRESS}th House: n={len(V)} members (>= {int(np.ceil(n_bills/2))} of {n_bills} votes)")


def wmean(x, w):
    ok = ~np.isnan(x)
    return np.average(x[ok], weights=w[ok] + 1e-12)


for K in (3, 4):
    pi = mixed_score(S, K, X, TAU0)
    pi = pi[:, np.argsort([wmean(nom, pi[:, k]) for k in range(K)])]       # order vertices by NOMINATE dim 1
    lab = []
    for k in range(K):
        d = np.average(party == "D", weights=pi[:, k] + 1e-12)
        near = ", ".join(mem.bioname.values[i].split(",")[0].title() for i in np.argsort(-pi[:, k])[:6])
        lab.append(f"V{k+1}\nD {d*100:.0f}% | NOM1 {wmean(nom, pi[:, k]):+.2f}")
        print(f"K={K} V{k+1}: D {d*100:3.0f}%  NOM1 {wmean(nom, pi[:, k]):+.2f}  nearest: {near}")
    title = f"{CONGRESS}th House, K={K}, weighted agreement S + speech PCs (tau=1), n={len(V)}"
    if K == 3:
        vt = np.array([[0, 0], [1, 0], [.5, np.sqrt(3) / 2]])
        xy = pi @ vt + rng.normal(scale=.003, size=(len(pi), 2))
        fig, ax = plt.subplots(figsize=(7.5, 6.8))
        ax.plot(*np.vstack([vt, vt[0]]).T, "k-", lw=1)
        ax.scatter(*xy.T, c=col, s=20, alpha=.65, edgecolor="white", lw=.3)
        for k, (x, y) in enumerate(vt):
            ax.text(x + (x - .5) * .12, y + (y - .29) * .12, lab[k], ha="center", va="center", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-.2, 1.2); ax.set_ylim(-.2, 1.05); ax.set_title(title, fontsize=10)
        ax.scatter([], [], c="#2166ac", label="Democrat"); ax.scatter([], [], c="#b2182b", label="Republican")
        ax.legend(frameon=False, loc="upper left")
        plt.tight_layout(); plt.savefig(HERE / "simplex_K3_118.png", dpi=140); plt.close()
    else:
        vt = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)    # regular tetrahedron
        P = pi @ vt
        fig = plt.figure(figsize=(14, 6.5))
        for j, (el, az) in enumerate(((20, 35), (20, 125))):
            ax = fig.add_subplot(1, 2, j + 1, projection="3d")
            for a, b in combinations(range(4), 2):
                ax.plot(*vt[[a, b]].T, "k-", lw=.9)
            ax.scatter(*P.T, c=col, s=18, alpha=.7)
            for k in range(4):
                ax.text(*(vt[k] * 1.25), lab[k], ha="center", fontsize=8)
            ax.set_axis_off(); ax.view_init(el, az); ax.set_box_aspect((1, 1, 1))
        fig.suptitle(title + " (two views)")
        plt.tight_layout(); plt.savefig(HERE / "simplex_K4_118.png", dpi=130); plt.close()
