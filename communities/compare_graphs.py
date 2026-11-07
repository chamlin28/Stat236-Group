import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as c, mixed_score as ms
from scipy.sparse.csgraph import connected_components
congress = 118
V, mem = c.load_votes(congress); S = c.agreement_matrix(V)
rng = np.random.default_rng(0)

def knn(S, m=40):
    n = len(S); T = S + rng.uniform(0, 1e-6, S.shape); A = np.zeros_like(S)
    for i in range(n):
        A[i, np.argsort(-T[i])[:m]] = 1
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0); return A

graphs = {"thresholded A": c.adjacency(S), "weighted S": S, "kNN (m=40)": knn(S)}
party = mem.party_code.map({100: "D", 200: "R"}).fillna("I").values
col = np.where(party == "D", "#2166ac", np.where(party == "R", "#b2182b", "#555"))
rows = []; store = {}
for gname, G in graphs.items():
    _, lab = connected_components(G > 0); keep = lab == np.bincount(lab).argmax()
    A = G[keep][:, keep]; m = mem[keep]; p = party[keep]
    Xdf = c.speech_covariates(congress, m.bioguide_id.values); X = Xdf.fillna(Xdf.mean()).values
    nom = m.nominate_dim1.fillna(0).values
    for K in (3, 4):
        for tau0 in (0.0, 1.0):
            r = ms.mixed_score(A, K, X, tau0); pi = r["pi"]
            Dsh = np.array([np.average(p == "D", weights=pi[:, k] + 1e-12) for k in range(K)])
            order = np.argsort([np.average(nom, weights=pi[:, k] + 1e-12) for k in range(K)])
            pi, Dsh = pi[:, order], Dsh[order]
            leak = np.mean([pi[i, (Dsh > .5) == (p[i] != "D")].sum() for i in range(len(p)) if p[i] in "DR"])
            rows.append(dict(graph=gname, n=keep.sum(), K=K, tau0=tau0, pure90=(pi.max(1) > .9).mean(),
                             cross_party_mass=leak, comm_D_share=np.round(Dsh, 2).tolist(),
                             comm_mean_pi=np.round(pi.mean(0), 2).tolist()))
            store[(gname, K, tau0)] = (pi, p)
df = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(df.round(2).to_string(index=False))
df.to_csv(c.ROOT / "communities/graph_comparison_118.csv", index=False)

def proj(K):
    return np.array([[0, 0], [1, 0], [.5, np.sqrt(3) / 2]]) if K == 3 else np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
for K in (3, 4):
    fig, axs = plt.subplots(2, 3, figsize=(13, 8.5 if K == 3 else 9))
    for j, gname in enumerate(graphs):
        for i, tau0 in enumerate((0.0, 1.0)):
            pi, p = store[(gname, K, tau0)]; ax = axs[i, j]; vt = proj(K)
            xy = pi @ vt + rng.normal(scale=.004, size=(len(pi), 2))
            ax.plot(*np.vstack([vt, vt[0]]).T, "k-", lw=.8)
            cc = np.where(p == "D", "#2166ac", np.where(p == "R", "#b2182b", "#555"))
            ax.scatter(*xy.T, c=cc, s=14, alpha=.6, edgecolor="none")
            ax.set_title(f"{gname}, tau={int(tau0)}", fontsize=10); ax.set_aspect("equal"); ax.axis("off")
            for k in range(K): ax.text(*(vt[k] + (vt[k] - vt.mean(0)) * .08), str(k + 1), ha="center", va="center", fontsize=9)
    fig.suptitle(f"{congress}th House, K={K}: Mixed-SCORE memberships (blue=D, red=R; vertices ordered by NOMINATE dim 1)"
                 + ("" if K == 3 else "\n(K=4 tetrahedron projected onto a square: distances are distorted)"), fontsize=11)
    plt.tight_layout(); plt.savefig(c.ROOT / f"communities/compare_K{K}_{congress}.png", dpi=110)
