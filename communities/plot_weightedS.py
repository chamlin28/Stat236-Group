import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as c, mixed_score as ms
congress, TAU0 = 118, 1.0
MINV = 180   # at least half of the 360 final-passage votes
V, mem = c.load_votes(congress, min_votes=MINV); S = c.agreement_matrix(V)
Xdf = c.speech_covariates(congress, mem.bioguide_id.values); X = Xdf.fillna(Xdf.mean()).values
party = mem.party_code.map({100: "D", 200: "R"}).fillna("I").values
nom = mem.nominate_dim1.fillna(0).values
col = np.where(party == "D", "#2166ac", np.where(party == "R", "#b2182b", "#555"))
rng = np.random.default_rng(0)
res = {}
for K in (3, 4):
    pi = ms.mixed_score(S, K, X, TAU0)["pi"]
    order = np.argsort([np.average(nom, weights=pi[:, k] + 1e-12) for k in range(K)]); pi = pi[:, order]
    res[K] = pi
    out = mem[["bioguide_id", "bioname", "state_abbrev"]].copy(); out["party"] = party
    for k in range(K): out[f"pi{k+1}"] = pi[:, k]
    out.to_csv(c.ROOT / f"communities/weightedS_K{K}_min180_tau1_memberships.csv", index=False)
    lab = [f"V{k+1}\nD {np.average(party=='D', weights=pi[:,k]+1e-12)*100:.0f}% | NOM1 {np.average(nom, weights=pi[:,k]+1e-12):+.2f}" for k in range(K)]
    if K == 3:
        vt = np.array([[0, 0], [1, 0], [.5, np.sqrt(3) / 2]]); xy = pi @ vt + rng.normal(scale=.003, size=(len(pi), 2))
        fig, ax = plt.subplots(figsize=(7.5, 6.8))
        ax.plot(*np.vstack([vt, vt[0]]).T, "k-", lw=1); ax.scatter(*xy.T, c=col, s=20, alpha=.65, edgecolor="white", lw=.3)
        for k, (x, y) in enumerate(vt): ax.text(x + (x - .5) * .12, y + (y - .29) * .12, lab[k], ha="center", va="center", fontsize=9)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-.2, 1.2); ax.set_ylim(-.2, 1.05)
        ax.set_title(f"{congress}th House, K=3, weighted agreement S, tau=1"); 
        ax.scatter([], [], c="#2166ac", label="Democrat"); ax.scatter([], [], c="#b2182b", label="Republican"); ax.legend(frameon=False, loc="upper left")
        plt.tight_layout(); plt.savefig(c.ROOT / "communities/weightedS_K3_min180_tau1_triangle.png", dpi=140)
    else:
        vt = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)   # regular tetrahedron
        P = pi @ vt
        fig = plt.figure(figsize=(14, 6.5))
        for j, (el, az) in enumerate(((20, 35), (20, 125))):
            ax = fig.add_subplot(1, 2, j + 1, projection="3d")
            for a, b in combinations(range(4), 2): ax.plot(*vt[[a, b]].T, "k-", lw=.9)
            ax.scatter(*P.T, c=col, s=18, alpha=.7, depthshade=True)
            for k in range(4): ax.text(*(vt[k] * 1.25), lab[k], ha="center", fontsize=8)
            ax.set_axis_off(); ax.view_init(el, az); ax.set_box_aspect((1, 1, 1))
        fig.suptitle(f"{congress}th House, K=4, weighted agreement S, tau=1 (two views of the same tetrahedron)")
        plt.tight_layout(); plt.savefig(c.ROOT / "communities/weightedS_K4_min180_tau1_tetrahedron.png", dpi=130)
for K, pi in res.items():
    print(f"K={K}: pure(>.9) {(pi.max(1)>.9).mean():.2f}; mean pi D:", np.round(pi[party=='D'].mean(0), 2), "R:", np.round(pi[party=='R'].mean(0), 2))
    for k in range(K):
        t = np.argsort(-pi[:, k])[:5]; print(f"  V{k+1} nearest:", "; ".join(f"{mem.bioname.values[i].split(',')[0]} ({party[i]})" for i in t))
