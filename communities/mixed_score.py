"""Mixed-SCORE (Jin, Ke, Luo) with node attributes (Ji & Jin style [R, tau*H] embedding).

1. SCORE:  R = [xi_2/xi_1, ..., xi_K/xi_1]  from top-K eigenvectors of A (ratios truncated at +-log n)
2. Attributes: H = top-K left singular vectors of the standardised covariate matrix X; embed Z = [R, tau*H]
3. Vertex hunting: successive projection algorithm (SPA) on the rows of Z
4. Barycentric coordinates w_i by simplex-constrained least squares (QP); then
   pi_i  proportional to  w_i / b1  (entry-wise),  b1(k) = [l_1 + v_k' diag(l_2..l_K) v_k]^(-1/2)
"""
import numpy as np
from scipy.optimize import nnls


def score_matrix(A, K):
    lam, U = np.linalg.eigh(A)
    idx = np.argsort(-np.abs(lam))[:K]
    lam, U = lam[idx], U[:, idx]
    U[:, 0] *= np.sign(U[:, 0].sum())                 # xi_1 positive
    n = A.shape[0]
    R = U[:, 1:] / U[:, [0]]
    T = np.log(n)
    return np.clip(R, -T, T), lam, U


def spa(Z, K):
    """Successive projection algorithm (Araujo et al. 2001), as in the lecture."""
    Y = Z.copy(); idx = []
    for _ in range(K):
        j = int(np.argmax((Y ** 2).sum(1))); idx.append(j)
        u = Y[j] / np.linalg.norm(Y[j])
        Y = Y - np.outer(Y @ u, u)
    return idx


def vertex_hunt(Z, K, n_centers=None, seed=0):
    """Denoised SPA: k-means to many centers (removes noise-driven outliers), SPA on centers."""
    from sklearn.cluster import KMeans
    L = n_centers or max(10 * K, 30)
    C = KMeans(L, n_init=10, random_state=seed).fit(Z).cluster_centers_
    scale = np.sqrt((C ** 2).sum(1).mean())
    Ch = np.hstack([np.full((L, 1), scale), C])      # homogeneous coords [1, z]: simplex need not contain 0
    return C[spa(Ch, K)]


def robust_sd(M):
    return 1.4826 * np.median(np.abs(M - np.median(M, 0)), 0)


def barycentric(Z, Vt, big=1e3):
    """argmin ||z - Vw||  s.t. w>=0, sum w = 1   (sum constraint as heavily weighted row)."""
    K = Vt.shape[0]
    M = np.vstack([Vt.T, big * np.ones((1, K))])
    W = np.zeros((Z.shape[0], K))
    for i, z in enumerate(Z):
        W[i], _ = nnls(M, np.concatenate([z, [big]]))
    return W / W.sum(1, keepdims=True)


def covariate_embedding(X, K):
    Xs = (X - X.mean(0)) / X.std(0)
    Uh, _, _ = np.linalg.svd(Xs, full_matrices=False)
    return Uh[:, :K]


def mixed_score(A, K, X=None, tau0=1.0):
    R, lam, U = score_matrix(A, K)
    Z = R
    tau = 0.0
    if X is not None and tau0 > 0:
        H = covariate_embedding(X, min(K, X.shape[1]))
        # tau0 = 1: covariate block has the same total (robust) spread as the graph block
        tau = tau0 * np.sqrt((robust_sd(R) ** 2).sum() / (robust_sd(H) ** 2).sum())
        Z = np.hstack([R, tau * H])
    Vt = vertex_hunt(Z, K)
    W = barycentric(Z, Vt)
    vR = Vt[:, :K - 1]
    b1 = (lam[0] + np.einsum('kd,d,kd->k', vR, lam[1:], vR)) ** -0.5
    P = W / b1
    P = P / P.sum(1, keepdims=True)
    return dict(pi=P, w=W, vertices_xy=Vt, R=R, Z=Z, lam=lam, tau=tau)
