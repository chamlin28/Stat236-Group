import sys, itertools, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import sgnq
rng = np.random.default_rng(0)
# 1) brute-force check of the 4-cycle sum
n = 9; M = rng.normal(size=(n, n)); M = (M + M.T) / 2
brute = sum(M[a, b] * M[b, c] * M[c, d] * M[d, a] for a, b, c, d in itertools.product(range(n), repeat=4) if len({a, b, c, d}) == 4)
print("brute force:", round(brute, 6), " fast:", round(sgnq.Qn_exact(M), 6))
# 2) calibration under H0 (K=1 DCBM, heterogeneous theta) and power under K=2
def sim(n, theta, Pm=None, pi=None):
    if Pm is None: Om = np.outer(theta, theta)
    else: Om = np.outer(theta, theta) * (pi @ Pm @ pi.T)
    U = rng.random((n, n)); A = np.triu((U < Om).astype(float), 1); return A + A.T
n = 600
Z0 = []
for r in range(200):
    th = rng.pareto(12, n) + 1; th = th / np.linalg.norm(th) * 12 / 1.0; th = np.minimum(th, 0.9 * 1 / th.max() * th.max())
    th = th / th.max() * 0.6 if th.max() > 0.6 else th
    Z0.append(sgnq.test(sim(n, th))["Z"])
print("null: mean Z %.2f sd Z %.2f  (paper: limiting N(1/sqrt2? ...) reported empirical mean ~0, sd ~0.92); reject rate at .05: %.3f" % (np.mean(Z0), np.std(Z0), np.mean(np.array(Z0) > 1.645)))
pi = np.zeros((n, 2)); pi[: n // 2, 0] = 1; pi[n // 2:, 1] = 1
Pm = np.array([[1, .5], [.5, 1]]); Z1 = []
for r in range(50):
    th = rng.pareto(12, n) + 1; th = th / th.max() * 0.6
    Z1.append(sgnq.test(sim(n, th, Pm, pi))["Z"])
print("alt (K=2): mean Z %.1f, reject rate at .001: %.2f" % (np.mean(Z1), np.mean([sgnq.test.__defaults__ and (z > 3.09) for z in Z1])))
