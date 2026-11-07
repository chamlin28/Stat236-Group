"""Build member-level speech features for the tilt models, from saved speech embeddings.

Usage:
  python speech_features.py                      # Congresses 115 116 118, PCA fit on 115 + 116
  python speech_features.py --congresses 115 116 117 118 --train 115 116

Reads  $CREC_DATA_DIR/embeddings/speech_emb_<N>.npy and speech_meta_<N>.parquet
       (written by speech_embeddings.py) and the group's data/elections/targets.csv (for party).
Writes $CREC_DATA_DIR/embeddings/speech_features.csv, one row per (congress, bioguide).

Columns
  Simple:     log_speeches, log_words, words_per_speech, share_extension, log_floor, log_ext
  Embeddings: for each subset (all, floor, ext) the first K principal components of the member
              mean embedding: <subset>_pc1..K (raw) and <subset>_pc1_pa..K_pa (party-adjusted).
Leakage control: PCA loadings and centering come from the TRAIN Congresses only. Party adjustment
subtracts the (congress, party) mean of each score within that Congress, which uses no outcomes.
Members with too few speeches in a subset get NaN for that subset (the notebook fills with 0).
"""
import argparse, os, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--congresses", type=int, nargs="+", default=[115, 116, 118])
ap.add_argument("--train", type=int, nargs="+", default=[115, 116])
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--min-all", type=int, default=5, help="min speeches for the overall embedding")
ap.add_argument("--min-subset", type=int, default=3, help="min speeches for floor / extension embeddings")
ap.add_argument("--embeddings-dir", default=None)
ap.add_argument("--targets", default=os.path.expanduser("~/Stat236-Group/data/elections/targets.csv"))
ap.add_argument("--out", default=None)
args = ap.parse_args()

base = os.environ.get("CREC_DATA_DIR", os.path.expanduser("~/crec_data"))
emb = args.embeddings_dir or os.path.join(base, "embeddings")
out_path = args.out or os.path.join(emb, "speech_features.csv")
assert set(args.train) <= set(args.congresses), "train Congresses must be among --congresses"
K = args.k

targets = pd.read_csv(args.targets)
party = targets[targets.party.isin(["D", "R"])].drop_duplicates(["congress", "bioguide"]) \
    .set_index(["congress", "bioguide"]).party

SUBSETS = {"all": (None, args.min_all), "floor": (False, args.min_subset), "ext": (True, args.min_subset)}

# ---- 1. Load and compute member means per subset ----
simple, means = [], {s: {} for s in SUBSETS}     # means[subset][congress] = (bioguides, X)
for c in args.congresses:
    S = np.load(os.path.join(emb, f"speech_emb_{c}.npy"))
    meta = pd.read_parquet(os.path.join(emb, f"speech_meta_{c}.parquet"))
    assert len(S) == len(meta), f"row mismatch in Congress {c}"
    w = meta.n_words.values.astype(np.float32)
    isext = meta.is_extension.values.astype(bool)

    g = meta.assign(ext=isext.astype(float)).groupby("bioguide")
    f = pd.DataFrame({"n": g.size(), "words": g.n_words.sum(), "share_extension": g.ext.mean(),
                      "n_ext": g.ext.sum()})
    f["n_floor"] = f.n - f.n_ext
    simple.append(pd.DataFrame({
        "congress": c, "bioguide": f.index,
        "log_speeches": np.log1p(f.n.values), "log_words": np.log1p(f.words.values),
        "words_per_speech": (f.words / f.n).values, "share_extension": f.share_extension.values,
        "log_floor": np.log1p(f.n_floor.values), "log_ext": np.log1p(f.n_ext.values)}))

    for sname, (flag, min_n) in SUBSETS.items():
        mask = np.ones(len(meta), bool) if flag is None else (isext == flag)
        pos = np.flatnonzero(mask)
        groups = pd.Series(pos).groupby(meta.bioguide.values[mask]).agg(list)
        ids, rows = [], []
        for b, idx in groups.items():
            if len(idx) < min_n:
                continue
            v = (S[idx] * w[idx, None]).sum(0)
            nv = np.linalg.norm(v)
            if nv == 0:
                continue
            ids.append(b); rows.append(v / nv)
        means[sname][c] = (np.array(ids), np.array(rows, dtype=np.float64))

# ---- 2. PCA on the training Congresses only, then project everything ----
def pca_fit(X, k):
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Vt = Vt[:k].copy()
    for j in range(k):                                    # fixed sign convention
        if Vt[j, np.argmax(np.abs(Vt[j]))] < 0:
            Vt[j] *= -1
    return mu, Vt, (s ** 2 / (s ** 2).sum())[:k]

feat_frames = []
print("MEMBERS WITH ENOUGH SPEECHES")
rows = []
for c in args.congresses:
    all_members = len(simple[args.congresses.index(c)])
    rows.append({"congress": c, "members_with_speeches": all_members,
                 **{f"valid_{s}": len(means[s][c][0]) for s in SUBSETS}})
print(pd.DataFrame(rows).to_string(index=False))

fits = {}
for sname in SUBSETS:
    Xtr = np.vstack([means[sname][c][1] for c in args.train])
    fits[sname] = pca_fit(Xtr, K)
    print(f"\n[{sname}] PCA fit on Congresses {args.train} ({len(Xtr)} member-rows). "
          f"Variance explained by PC1-{K}: {[round(float(v), 3) for v in fits[sname][2]]}")

for sname in SUBSETS:
    mu, Vt, _ = fits[sname]
    for c in args.congresses:
        ids, X = means[sname][c]
        sc = (X - mu) @ Vt.T
        df = pd.DataFrame(sc, columns=[f"{sname}_pc{j+1}" for j in range(K)])
        df["bioguide"] = ids; df["congress"] = c
        p = party.reindex(pd.MultiIndex.from_arrays([[c] * len(ids), ids])).values
        for j in range(K):
            col = f"{sname}_pc{j+1}"
            s_ = pd.Series(df[col].values)
            adj = s_ - s_.groupby(p).transform("mean")    # NaN party stays NaN
            df[col + "_pa"] = adj.values
        feat_frames.append(df)

emb_df = pd.concat(feat_frames)
emb_df = emb_df.groupby(["congress", "bioguide"]).first().reset_index()
out = pd.concat(simple).merge(emb_df, on=["congress", "bioguide"], how="left")
out = out[out.congress.isin(args.congresses)]
out.to_csv(out_path, index=False)
print(f"\nWrote {out_path}: {len(out)} rows, {out.shape[1] - 2} feature columns")

# ---- 3. Diagnostics ----
print("\nSTABILITY: |cosine| between each Congress's own PC directions and the training directions (subset 'all')")
rows = []
for c in args.congresses:
    own_mu, own_Vt, _ = pca_fit(means["all"][c][1], 3)
    rows.append({"congress": c, **{f"PC{j+1}": round(float(abs(own_Vt[j] @ fits['all'][1][j])), 2) for j in range(3)}})
print(pd.DataFrame(rows).to_string(index=False))

print("\nCORRELATION of all_pc1..3 (raw) with share_extension, by Congress")
rows = []
for c in args.congresses:
    d = out[out.congress == c].dropna(subset=["all_pc1"])
    rows.append({"congress": c, **{f"PC{j+1}": round(float(np.corrcoef(d[f'all_pc{j+1}'], d.share_extension)[0, 1]), 2) for j in range(3)}})
print(pd.DataFrame(rows).to_string(index=False))

print("\nCORRELATION of all_pc1..3 with party (D=1), raw vs party-adjusted")
rows = []
for c in args.congresses:
    d = out[out.congress == c].dropna(subset=["all_pc1"])
    pr = party.reindex(pd.MultiIndex.from_arrays([[c] * len(d), d.bioguide])).values
    ok = ~pd.isna(pr); y = (pr[ok] == "D").astype(float)
    r = {"congress": c}
    for j in range(3):
        r[f"raw_PC{j+1}"] = round(float(abs(np.corrcoef(d[f"all_pc{j+1}"].values[ok], y)[0, 1])), 2)
        r[f"pa_PC{j+1}"] = round(float(abs(np.corrcoef(d[f"all_pc{j+1}_pa"].values[ok], y)[0, 1])), 2)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
