import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as c, mixed_score as ms
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linear_sum_assignment
V, mem = c.load_votes(118); S = c.agreement_matrix(V); A = c.adjacency(S)
_, lab = connected_components(A); keep = lab == np.bincount(lab).argmax()
A, mem = A[keep][:, keep], mem[keep]
Xdf = c.speech_covariates(118, mem.bioguide_id.values); X = Xdf.fillna(Xdf.mean()).values
party = (mem.party_code.values == 200).astype(int)
for K in (3, 4):
    base = ms.mixed_score(A, K, X, 0.0)["pi"]
    for t in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        pi = ms.mixed_score(A, K, X, t)["pi"]
        C = np.corrcoef(base.T, pi.T)[:K, K:]; r, cidx = linear_sum_assignment(-C)
        pur = np.mean([max(party[pi.argmax(1) == k].mean(), 1 - party[pi.argmax(1) == k].mean()) for k in range(K) if (pi.argmax(1) == k).any()])
        print(f"K={K} tau0={t:<4} corr w/ graph-only pi: {C[r, cidx].mean():.2f}   max-pi share>.9: {(pi.max(1)>.9).mean():.2f}  party purity of argmax: {pur:.2f}")
