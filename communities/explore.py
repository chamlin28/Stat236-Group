import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as c, mixed_score as ms
congress = 118
V, mem = c.load_votes(congress)
S = c.agreement_matrix(V); A = c.adjacency(S)
from scipy.sparse.csgraph import connected_components
ncomp, lab = connected_components(A)
keep = lab == np.bincount(lab).argmax()
print('components:', ncomp, ' giant size:', keep.sum())
A = A[keep][:, keep]; mem = mem[keep]; V = V[keep]
Xdf = c.speech_covariates(congress, mem.bioguide_id.values)
X = Xdf.fillna(Xdf.mean()).values
print("n after dropping isolated:", len(A))
for K in (2, 3, 4):
    for tau0 in (0.0, 0.5, 1.0):
        r = ms.mixed_score(A, K, X, tau0)
        pi = r["pi"]
        d = ((r["Z"][:, None, :] - r["vertices_xy"][None]) ** 2).sum(2).argmin(0)
        vtx = mem.bioname.values[d]
        tab = pd.DataFrame(pi, columns=[f"c{k+1}" for k in range(K)]); tab["party"] = mem.party_code.values
        print(f"\nK={K} tau0={tau0} vertex members: {list(vtx)}")
        print(tab.groupby("party").mean().round(2).to_string())
        print("share of members with max pi > .9:", round((pi.max(1) > .9).mean(), 2))
