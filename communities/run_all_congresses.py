import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as c, mixed_score as ms
OUT = c.ROOT / "communities/by_congress"; OUT.mkdir(exist_ok=True)
rows = []; summary = []

def wmean(x, w):
    ok = ~np.isnan(x); return np.average(x[ok], weights=w[ok] + 1e-12)

for congress in (115, 116, 117, 118, 119):
    nb = len(pd.read_csv(c.ROOT / f"data/{congress}/bills.csv"))
    V, mem = c.load_votes(congress, min_votes=int(np.ceil(nb / 2)))
    S = c.agreement_matrix(V)
    Xdf = c.speech_covariates(congress, mem.bioguide_id.values)
    has_x = Xdf.notna().any(axis=1).mean() > .5
    if not has_x:
        print(f"\n=== {congress}th: SKIPPED (no speech features, tau=1 not possible)"); continue
    tau0 = 1.0
    X = Xdf.fillna(Xdf.mean()).values
    party = mem.party_code.map({100: "D", 200: "R"}).fillna("I").values
    nom = mem.nominate_dim1.values.astype(float)
    col = np.where(party == "D", "#2166ac", np.where(party == "R", "#b2182b", "#555"))
    rng = np.random.default_rng(0)
    print(f"\n=== {congress}th: {nb} bills, cutoff {int(np.ceil(nb/2))} votes, n={len(V)} members, tau={tau0} ({'with' if has_x else 'NO'} speech covariates)")
    for K in (3, 4):
        pi = ms.mixed_score(S, K, X, tau0)["pi"]
        order = np.argsort([wmean(nom, pi[:, k]) for k in range(K)]); pi = pi[:, order]
        out = mem[["bioguide_id", "bioname", "state_abbrev"]].copy(); out["party"] = party
        for k in range(K): out[f"pi{k+1}"] = pi[:, k]
        out.to_csv(OUT / f"{congress}_K{K}_memberships.csv", index=False)
        lab = []
        for k in range(K):
            d = np.average(party == "D", weights=pi[:, k] + 1e-12); n1 = wmean(nom, pi[:, k])
            near = ", ".join(mem.bioname.values[i].split(",")[0].title() + f" ({party[i]}, {nom[i]:+.2f})" for i in np.argsort(-pi[:, k])[:8])
            lab.append(f"V{k+1}\nD {d*100:.0f}% | NOM1 {n1:+.2f}")
            print(f"  K={K} V{k+1}: D {d*100:3.0f}%  NOM1 {n1:+.2f}  nearest: {near}")
            summary.append(dict(congress=congress, K=K, vertex=k + 1, D_share=round(d, 2), nom1=round(n1, 2), nearest=near))
        print(f"  K={K} pure(>.9): {(pi.max(1)>.9).mean():.2f}")
        ttl = f"{congress}th House, K={K}, weighted agreement S, " + ("tau=1" if has_x else "tau=0 (no speech features)") + f", n={len(V)}"
        if K == 3:
            vt = np.array([[0, 0], [1, 0], [.5, np.sqrt(3) / 2]]); xy = pi @ vt + rng.normal(scale=.003, size=(len(pi), 2))
            fig, ax = plt.subplots(figsize=(7.5, 6.8))
            ax.plot(*np.vstack([vt, vt[0]]).T, "k-", lw=1); ax.scatter(*xy.T, c=col, s=20, alpha=.65, edgecolor="white", lw=.3)
            for k, (x, y) in enumerate(vt): ax.text(x + (x - .5) * .12, y + (y - .29) * .12, lab[k], ha="center", va="center", fontsize=9)
            ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-.2, 1.2); ax.set_ylim(-.2, 1.05); ax.set_title(ttl, fontsize=10)
            ax.scatter([], [], c="#2166ac", label="Democrat"); ax.scatter([], [], c="#b2182b", label="Republican"); ax.legend(frameon=False, loc="upper left")
            plt.tight_layout(); plt.savefig(OUT / f"{congress}_K3_triangle.png", dpi=120); plt.close()
        else:
            vt = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float); P = pi @ vt
            fig = plt.figure(figsize=(14, 6.5))
            for j, (el, az) in enumerate(((20, 35), (20, 125))):
                ax = fig.add_subplot(1, 2, j + 1, projection="3d")
                for a, b in combinations(range(4), 2): ax.plot(*vt[[a, b]].T, "k-", lw=.9)
                ax.scatter(*P.T, c=col, s=18, alpha=.7)
                for k in range(4): ax.text(*(vt[k] * 1.25), lab[k], ha="center", fontsize=8)
                ax.set_axis_off(); ax.view_init(el, az); ax.set_box_aspect((1, 1, 1))
            fig.suptitle(ttl + " (two views)"); plt.tight_layout(); plt.savefig(OUT / f"{congress}_K4_tetrahedron.png", dpi=110); plt.close()
pd.DataFrame(summary).to_csv(OUT / "vertex_summary.csv", index=False)
