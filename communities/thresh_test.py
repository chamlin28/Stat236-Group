import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as c, mixed_score as ms
from scipy.sparse.csgraph import connected_components
V, mem = c.load_votes(118); S = c.agreement_matrix(V)
party = (mem.party_code.values == 200).astype(int)
off = S[~np.eye(len(S), dtype=bool)]
print("agreement quantiles:", np.round(np.quantile(off, [.1, .25, .5, .75, .9]), 3), "mean+sd", round(off.mean() + off.std(), 3))
for thr in (0.80, 0.85, 0.90, 0.93, 0.95):
    A = (S > thr).astype(float); np.fill_diagonal(A, 0)
    _, lab = connected_components(A); keep = lab == np.bincount(lab).argmax()
    Ak = A[keep][:, keep]; p = party[keep]
    ev = np.sort(np.linalg.eigvalsh(Ak))[::-1]
    print(f"\nthr={thr} n={keep.sum()} dens={Ak.mean():.2f} ev={np.round(ev[:7],0)}")
    for K in (3, 4):
        try:
            pi = ms.mixed_score(Ak, K)["pi"]
            print(f"  K={K}: share pure(>.9)={(pi.max(1)>.9).mean():.2f}; mean pi by party D:", np.round(pi[p == 0].mean(0), 2), "R:", np.round(pi[p == 1].mean(0), 2))
        except Exception as e: print("  K", K, "failed", e)
