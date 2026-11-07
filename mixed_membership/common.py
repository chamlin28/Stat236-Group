"""Loaders: signed House vote matrix, pairwise agreement matrix S, speech covariates."""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_votes(congress, min_votes):
    """Members x final-passage bills: +1 Yea, -1 Nay, 0 absent/not voting.
    Keeps House members with at least `min_votes` Yea/Nay votes."""
    bills = pd.read_csv(ROOT / f"data/{congress}/bills.csv")
    mem = pd.read_csv(ROOT / f"data/{congress}/voteview/H{congress}_members.csv")
    mem = mem[mem.chamber == "House"].drop_duplicates("icpsr")
    v = pd.read_csv(ROOT / f"data/{congress}/voteview/H{congress}_votes.csv")
    v = v[v.chamber == "House"].merge(bills[["rollnumber", "bill_number"]], on="rollnumber")
    v = v[v.icpsr.isin(mem.icpsr)]
    v["y"] = np.where(v.cast_code.between(1, 3), 1, np.where(v.cast_code.between(4, 6), -1, 0))
    V = v.pivot_table(index="icpsr", columns="bill_number", values="y", aggfunc="first").fillna(0)
    V = V[(V != 0).sum(axis=1) >= min_votes]
    return V, mem.set_index("icpsr").loc[V.index]


def agreement_matrix(V):
    """S_ij = share of bills on which both i and j voted where they voted the same way; diagonal 0."""
    Vv = V.values
    voted = (Vv != 0).astype(float)
    yea, nay = (Vv == 1).astype(float), (Vv == -1).astype(float)
    S = (yea @ yea.T + nay @ nay.T) / np.maximum(voted @ voted.T, 1)
    np.fill_diagonal(S, 0)
    return S


def speech_covariates(congress, bioguides, cols=("all_pc1", "all_pc2", "all_pc3", "all_pc4", "all_pc5")):
    """Member-level speech-embedding principal components (data/speech/speech_features.csv)."""
    s = pd.read_csv(ROOT / "data/speech/speech_features.csv")
    s = s[s.congress == congress].drop_duplicates("bioguide").set_index("bioguide")
    return s.reindex(bioguides)[list(cols)]
