import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as c
cong = (115, 116, 118)
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for j, congress in enumerate(cong):
    nb = len(pd.read_csv(c.ROOT / f"data/{congress}/bills.csv"))
    V, mem = c.load_votes(congress, min_votes=int(np.ceil(nb / 2)))
    S = c.agreement_matrix(V)
    ev = np.sort(np.linalg.eigvalsh(S))[::-1]
    print(f"\n{congress}th (n={len(S)}): top 10 eigenvalues of S:", np.round(ev[:10], 2))
    print("   ratios l_k/l_(k+1):", np.round(ev[:9] / ev[1:10], 2))
    print("   share of total positive eigenvalue mass, top 1/2/3/4/5:", np.round(np.cumsum(ev[:5]) / ev[ev > 0].sum(), 3))
    ax[0, j].plot(range(1, 21), ev[:20], "o-"); ax[0, j].set_title(f"{congress}th House (n={len(S)})")
    ax[1, j].plot(range(2, 13), ev[1:12], "o-", c="C1"); ax[1, j].set_title("zoom: eigenvalues 2-12")
    for a in ax[:, j]: a.set_xlabel("k"); a.axvline(3, color="grey", ls=":", lw=.8)
ax[0, 0].set_ylabel("eigenvalue of S"); ax[1, 0].set_ylabel("eigenvalue of S")
fig.suptitle("Scree plots, weighted agreement matrix S (dotted line: k=3)")
plt.tight_layout(); plt.savefig(c.ROOT / "communities/scree_weightedS_115_116_118.png", dpi=120)
