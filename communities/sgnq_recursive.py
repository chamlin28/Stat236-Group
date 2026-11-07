"""Recursive community detection as in the MADStat co-authorship example (Lec10 supp, slide 17):
test K=1 with SgnQ; if p<0.001, split with SCORE (K=2: k-means on xi2/xi1) and repeat inside each piece
(restricted to its giant component)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as c, sgnq, mixed_score as ms
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
rng = np.random.default_rng(0)
MIN_N, ALPHA = 30, 0.001

def knn(S, m=40):
    n = len(S); T = S + rng.uniform(0, 1e-6, S.shape); A = np.zeros_like(S)
    for i in range(n): A[i, np.argsort(-T[i])[:m]] = 1
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0); return A

def giant(A, idx):
    _, lab = connected_components(A); keep = lab == np.bincount(lab).argmax()
    return A[keep][:, keep], idx[keep]

def split(A, idx, party, name, path, out, depth=0):
    if depth == 0:   # disconnected components are separate top-level communities
        ncomp, lab = connected_components(A)
        pieces = [np.where(lab == g)[0] for g in range(ncomp)]
        pieces = [q for q in pieces if len(q) >= MIN_N]
        print(f"  {ncomp} connected components; {len(pieces)} with n>={MIN_N}: sizes {[len(q) for q in pieces]}, "
              f"D share {[round(float((party[idx[q]]=='D').mean()),2) for q in pieces]}")
        for g, q in enumerate(pieces):
            split(A[q][:, q], idx[q], party, name, f"c{g}", out, 1)
        return
    A, idx = giant(A, idx)
    if len(idx) < MIN_N:
        out.append((path, len(idx), None, None, party[idx])); print(f"{'  '*depth}{path}: n={len(idx)} too small, stop"); return
    r = sgnq.test(A, ALPHA)
    pd_ = party[idx]; share = (pd_ == "D").mean()
    print(f"{'  '*depth}{path}: n={len(idx)} D={share:.0%}  Z={r['Z']:.1f} p={r['p']:.2g} -> {'SPLIT' if r['reject'] else 'stop (K=1 not rejected)'}")
    if not r["reject"]:
        out.append((path, len(idx), r["Z"], r["p"], pd_)); return
    R, lam, U = ms.score_matrix(A, 2)
    lab = KMeans(2, n_init=10, random_state=0).fit_predict(R)
    for g in (0, 1):
        sel = lab == g
        split(A[sel][:, sel], idx[sel], party, name, path + str(g), out, depth + 1)

for congress in (115, 116, 118):
    nb = len(pd.read_csv(c.ROOT / f"data/{congress}/bills.csv"))
    V, mem = c.load_votes(congress, min_votes=int(np.ceil(nb / 2))); S = c.agreement_matrix(V)
    party = mem.party_code.map({100: "D", 200: "R"}).fillna("I").values
    for gname, A in (("thresholded A (mean+sd)", c.adjacency(S)), ("kNN m=40", knn(S))):
        print(f"\n=== {congress}th, {gname}; n={len(S)}, density={A.sum()/(len(A)*(len(A)-1)):.2f}")
        out = []; split(A, np.arange(len(A)), party, gname, "root", out)
        print(f"  -> {len(out)} leaf communities")
