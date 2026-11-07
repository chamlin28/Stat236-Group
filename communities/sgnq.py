"""SgnQ test (Jin, Ke, Luo 2019, 'Optimal adaptivity of signed-polygon statistics for network testing').
H0: K=1 (DCMM with one community)  vs  H1: K>=2.  Needs a BINARY symmetric adjacency matrix.
  eta_hat = A 1 / sqrt(1'A1);   A~ = A - eta eta';   Q_n = sum over distinct (i1..i4) of the 4-cycle product of A~
  Z = [Q_n - 2(|eta|^2 - 1)^2] / sqrt(8 (|eta|^2 - 1)^4),  one-sided: reject H0 for large Z.
"""
import numpy as np
from scipy.stats import norm


def Qn_exact(At):
    """Sum over distinct (i1,i2,i3,i4) of At12 At23 At34 At41 via inclusion-exclusion on index coincidences.
    Validated against brute force in the test script."""
    n = len(At); B = At.copy(); np.fill_diagonal(B, 0)       # distinct i1!=i2 etc. only ever uses off-diagonal entries
    B2 = B @ B; B3 = B2 @ B; B4 = B3 @ B
    # closed 4-walks with i1!=i2, i2!=i3, i3!=i4, i4!=i1 (B has zero diagonal): tr(B^4)
    # remove those with i1=i3 and/or i2=i4:
    #   i1=i3 only-or-both: sum_{i1,i2,i4} B_{i1i2}B_{i2i1}B_{i1i4}B_{i4i1} = sum_i (sum_j B_ij^2)^2
    #   i2=i4: same by symmetry;  both: sum_{i,j} B_ij^4
    s2 = (B ** 2).sum(1)
    both = (B ** 4).sum()
    return np.trace(B4) - 2 * (s2 ** 2).sum() + both


def test(A, alpha=0.001):
    n = len(A); d = A.sum(1); V = d.sum()
    eta = d / np.sqrt(V)
    At = A - np.outer(eta, eta)
    Q = Qn_exact(At)
    e2 = (eta ** 2).sum() - 1
    Z = (Q - 2 * e2 ** 2) / np.sqrt(8 * e2 ** 4)
    return dict(Q=Q, Z=Z, p=norm.sf(Z), reject=norm.sf(Z) < alpha, norm_eta2=e2)
