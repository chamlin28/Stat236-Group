import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as c, mixed_score as ms
from scipy.sparse.csgraph import connected_components
congress, K = 118, 3
TAU0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
TAG = "" if TAU0 == 0.1 else f"_tau{TAU0}"
V, mem = c.load_votes(congress); S = c.agreement_matrix(V); A = c.adjacency(S)
_, lab = connected_components(A); keep = lab == np.bincount(lab).argmax()
A, mem = A[keep][:, keep], mem[keep].copy()
Xdf = c.speech_covariates(congress, mem.bioguide_id.values); X = Xdf.fillna(Xdf.mean()).values
r = ms.mixed_score(A, K, X, TAU0); pi = r["pi"]

# order communities by mean NOMINATE dim1 so labels are stable (left -> right)
order = np.argsort([np.average(mem.nominate_dim1.fillna(0), weights=pi[:, k]) for k in range(K)])
pi = pi[:, order]
mem["party"] = mem.party_code.map({100: "D", 200: "R"}).fillna("I")
out = mem[["bioguide_id", "bioname", "state_abbrev", "party", "nominate_dim1", "nominate_dim2"]].copy()
for k in range(K): out[f"pi{k+1}"] = pi[:, k]
out.to_csv(c.ROOT / f"communities/memberships_{congress}{TAG}.csv", index=False)

print(f"n={len(out)} (giant component), K={K}, tau0={TAU0}, tau={r['tau']:.3f}")
prof = pd.DataFrame({f"community {k+1}": {
    "mean pi": pi[:, k].mean(),
    "share D (pi-weighted)": np.average(out.party == "D", weights=pi[:, k]),
    "mean NOMINATE dim1": np.average(out.nominate_dim1.fillna(0), weights=pi[:, k]),
    "mean NOMINATE dim2": np.average(out.nominate_dim2.fillna(0), weights=pi[:, k]),
    "n with pi>.9": int((pi[:, k] > .9).sum())} for k in range(K)}).round(2)
print(prof.to_string())
for k in range(K):
    t = out.sort_values(f"pi{k+1}", ascending=False).head(6)
    print(f"\ncommunity {k+1} top members:", "; ".join(f"{b.split(',')[0]} ({p}-{s})" for b, p, s in zip(t.bioname, t.party, t.state_abbrev)))

# ---- ternary plot (barycentric coordinates = pi, as in the lecture's simplex picture)
verts = np.array([[0, 0], [1, 0], [.5, np.sqrt(3) / 2]])      # community 1, 2, 3
xy = pi @ verts
col = out.party.map({"D": "#2166ac", "R": "#b2182b", "I": "#555"}).values
fig, ax = plt.subplots(figsize=(7.5, 6.6))
ax.fill(*verts.T, color="#f4f4f4", zorder=0); ax.plot(*np.vstack([verts, verts[0]]).T, "k-", lw=1)
for g in (.25, .5, .75):
    for a, b in ((0, 1), (1, 2), (2, 0)):
        p0 = np.zeros(3); p0[a] = g; p0[b] = 1 - g; p1 = np.zeros(3); p1[a] = g; p1[3 - a - b] = 1 - g
        ax.plot(*np.vstack([p0 @ verts, p1 @ verts]).T, color="#ddd", lw=.6, zorder=1)
rng = np.random.default_rng(0); jit = rng.normal(scale=.004, size=xy.shape)
ax.scatter(*(xy + jit).T, c=col, s=22, alpha=.65, edgecolor="white", lw=.3, zorder=3)
for k in range(K):
    i = np.argmax(pi[:, k]); ax.annotate(out.bioname.iloc[i].split(",")[0], xy[i], fontsize=7, zorder=4,
                                          xytext=(5, 5), textcoords="offset points")
names = [f"Community {k+1}\n(D {prof.iloc[1, k]*100:.0f}%, NOM1 {prof.iloc[2, k]:+.2f})" for k in range(K)]
ax.text(-.02, -.05, names[0], ha="center", va="top", fontsize=9); ax.text(1.02, -.05, names[1], ha="center", va="top", fontsize=9)
ax.text(.5, np.sqrt(3) / 2 + .03, names[2], ha="center", va="bottom", fontsize=9)
ax.scatter([], [], c="#2166ac", label="Democrat"); ax.scatter([], [], c="#b2182b", label="Republican"); ax.legend(loc="upper left", frameon=False)
ax.set_aspect("equal"); ax.axis("off"); ax.set_ylim(-.2, .98)
ax.set_title(f"{congress}th House: estimated mixed memberships on the simplex\n(Mixed-SCORE on vote-agreement network + speech covariates)", fontsize=10)
plt.tight_layout(); plt.savefig(c.ROOT / f"communities/simplex_{congress}{TAG}.png", dpi=140)
