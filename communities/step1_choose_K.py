import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import common as c
congress = int(sys.argv[1]) if len(sys.argv) > 1 else 118
V, mem = c.load_votes(congress)
S = c.agreement_matrix(V); A = c.adjacency(S)
print("n =", len(V), " edges =", int(A.sum() / 2), " density =", round(A.sum() / (len(A) * (len(A) - 1)), 3))
print("components / isolated nodes:", int((A.sum(1) == 0).sum()))
ev = np.sort(np.linalg.eigvalsh(A))[::-1]
print("top 10 eigenvalues of A:", np.round(ev[:10], 1))
print("eigengap ratios l_k/l_{k+1}:", np.round(ev[:9] / ev[1:10], 2))
# also on the weighted agreement matrix
evs = np.sort(np.linalg.eigvalsh(S))[::-1]; print("top eigenvalues of S:", np.round(evs[:10], 1))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(range(1, 21), ev[:20], "o"); ax[0].set_title(f"Scree: adjacency A ({congress}th House)")
ax[1].plot(range(1, 21), evs[:20], "o", c="C1"); ax[1].set_title("Scree: weighted agreement S")
for a in ax: a.set_xlabel("k"); a.set_ylabel("eigenvalue")
plt.tight_layout(); plt.savefig(c.ROOT / f"communities/scree_{congress}.png", dpi=130)
