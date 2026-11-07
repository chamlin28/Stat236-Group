"""Mixed-SCORE (Jin, Ke, Luo) with node attributes (Ji & Jin: embed rows of [R, tau*H]).

1. SCORE: R = [xi_2/xi_1, ..., xi_K/xi_1] from the top-K eigenvectors (by |eigenvalue|) of the matrix.
2. Attributes: H = top-K left singular vectors of the standardised covariates; Z = [R, tau*H].
3. Vertex hunting: k-means to L = max(10K, 30) centres (denoise), then SPA on [1, centre] (SVS*-style).
4. Barycentric coordinates w_i by simplex-constrained least squares; pi_i proportional to w_i / b_1,
   b_1(k) = [lambda_1 + v_k' diag(lambda_2..lambda_K) v_k]^(-1/2).
"""
import numpy as np
from scipy.optimize import nnls
from sklearn.cluster import KMeans


def score_matrix(A, K):
    lam, U = np.linalg.eigh(A)
    idx = np.argsort(-np.abs(lam))[:K]
    lam, U = lam[idx], U[:, idx]
    U[:, 0] *= np.sign(U[:, 0].sum())                       # xi_1 > 0 (Perron-Frobenius)
    T = np.log(A.shape[0])
    return np.clip(U[:, 1:] / U[:, [0]], -T, T), lam        # ratios truncated at +-log n


def spa(Z, K):
    """Successive projection algorithm."""
    Y, idx = Z.copy(), []
    for _ in range(K):
        j = int(np.argmax((Y ** 2).sum(1))); idx.append(j)
        u = Y[j] / np.linalg.norm(Y[j])
        Y = Y - np.outer(Y @ u, u)
    return idx


def vertex_hunt(Z, K, seed=0):
    L = max(10 * K, 30)
    C = KMeans(L, n_init=10, random_state=seed).fit(Z).cluster_centers_
    scale = np.sqrt((C ** 2).sum(1).mean())
    return C[spa(np.hstack([np.full((L, 1), scale), C]), K)]   # homogeneous coords: simplex need not contain 0


def barycentric(Z, V, big=1e3):
    """argmin ||z - V'w||, w >= 0, sum(w) = 1, for every row z."""
    M = np.vstack([V.T, big * np.ones((1, V.shape[0]))])
    W = np.array([nnls(M, np.concatenate([z, [big]]))[0] for z in Z])
    return W / W.sum(1, keepdims=True)


def robust_sd(M):
    return 1.4826 * np.median(np.abs(M - np.median(M, 0)), 0)


def mixed_score(A, K, X=None, tau0=1.0):
    """A: symmetric weighted/binary matrix; X: n x p covariates (or None). Returns pi (n x K)."""
    R, lam = score_matrix(A, K)
    Z = R
    if X is not None and tau0 > 0:
        Xs = (X - X.mean(0)) / X.std(0)
        H = np.linalg.svd(Xs, full_matrices=False)[0][:, :min(K, X.shape[1])]
        # tau0 = 1: covariate block has the same total robust spread as the graph block
        tau = tau0 * np.sqrt((robust_sd(R) ** 2).sum() / (robust_sd(H) ** 2).sum())
        Z = np.hstack([R, tau * H])
    V = vertex_hunt(Z, K)
    W = barycentric(Z, V)
    b1 = (lam[0] + np.einsum('kd,d,kd->k', V[:, :K - 1], lam[1:], V[:, :K - 1])) ** -0.5
    P = W / b1
    return P / P.sum(1, keepdims=True)
