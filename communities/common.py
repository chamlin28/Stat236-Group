"""Shared loaders: House agreement network (from Yea/Nay votes) and speech covariates."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def load_votes(congress, min_votes=40):
    """Signed member x bill matrix on House final-passage roll calls: +1 Yea, -1 Nay, 0 absent."""
    bills = pd.read_csv(ROOT / f"data/{congress}/bills.csv")
    mem = pd.read_csv(ROOT / f"data/{congress}/voteview/H{congress}_members.csv")
    mem = mem[mem.chamber == "House"].drop_duplicates("icpsr")
    v = pd.read_csv(ROOT / f"data/{congress}/voteview/H{congress}_votes.csv")
    v = v[v.chamber == "House"].merge(bills[["rollnumber", "bill_number"]], on="rollnumber")
    v = v[v.icpsr.isin(mem.icpsr)]
    v["y"] = np.where(v.cast_code.between(1, 3), 1, np.where(v.cast_code.between(4, 6), -1, 0))
    V = v.pivot_table(index="icpsr", columns="bill_number", values="y", aggfunc="first").fillna(0)
    V = V[(V != 0).sum(axis=1) >= min_votes]
    mem = mem.set_index("icpsr").loc[V.index]
    return V, mem


def agreement_matrix(V):
    """S_ij = share of bills both voted on where i and j voted the same way; diag 0."""
    Vv = V.values
    both = (Vv != 0).astype(float)
    n_both = both @ both.T
    agree = ((Vv == 1).astype(float) @ (Vv == 1).T.astype(float)
             + (Vv == -1).astype(float) @ (Vv == -1).T.astype(float))
    S = agree / np.maximum(n_both, 1)
    np.fill_diagonal(S, 0)
    return S


def adjacency(S, rule="mean+sd"):
    """Binary network as in the trade-network example: edge if S_ij > mu + sigma (off-diagonal)."""
    n = S.shape[0]
    off = S[~np.eye(n, dtype=bool)]
    A = (S > off.mean() + off.std()).astype(float)
    np.fill_diagonal(A, 0)
    return A


def speech_covariates(congress, bioguides, cols=("all_pc1", "all_pc2", "all_pc3", "all_pc4", "all_pc5")):
    s = pd.read_csv(ROOT / "data/speech/speech_features.csv")
    s = s[s.congress == congress].drop_duplicates("bioguide").set_index("bioguide")
    X = s.reindex(bioguides)[list(cols)]
    return X
