"""See what the first principal component of the member embeddings is picking up.

Usage: python pc_inspect.py 118 --pilot     (uses the pilot_ files)
       python pc_inspect.py 118             (uses the full-run files)
Prints how PC1-3 relate to speech volume and length, then shows example speeches
from the two ends of PC1.
"""
import argparse, os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("congress", type=int)
ap.add_argument("--pilot", action="store_true")
ap.add_argument("--min-n", type=int, default=None)
ap.add_argument("--examples", type=int, default=4)
args = ap.parse_args()
c = args.congress
base = os.environ.get("CREC_DATA_DIR", os.path.expanduser("~/crec_data"))
emb = os.path.join(base, "embeddings")
pre = "pilot_" if args.pilot else ""
min_n = args.min_n or (2 if args.pilot else 5)

M = np.load(os.path.join(emb, f"{pre}member_emb_{c}.npy"))
mm = pd.read_csv(os.path.join(emb, f"{pre}member_meta_{c}.csv"))
S = np.load(os.path.join(emb, f"{pre}speech_emb_{c}.npy"))
meta = pd.read_parquet(os.path.join(emb, f"{pre}speech_meta_{c}.parquet"))

ok = (mm.n_speeches >= min_n).values
X = M[ok]; mu = X.mean(0); Xc = X - mu
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
var = s ** 2 / (s ** 2).sum()
sc = U * s
info = mm[ok].reset_index(drop=True).copy()
info["words_per_speech"] = info.n_words / info.n_speeches
print(f"{ok.sum()} members with >= {min_n} speeches; variance explained by PC1-5:",
      [round(float(v), 3) for v in var[:5]])
print("\nCorrelation of each PC with member-level features:")
rows = {}
for k in range(3):
    rows[f"PC{k+1}"] = {col: round(float(np.corrcoef(sc[:, k], info[col])[0, 1]), 2)
                        for col in ["n_speeches", "n_words", "words_per_speech", "share_extension"]}
print(pd.DataFrame(rows).T.to_string())

z = (S - mu) @ Vt[0]
text = pd.read_parquet(os.path.join(base, "parsed", f"crec_{c}.parquet"), columns=["speech"]).speech
order = np.argsort(z)
def show(idx, title):
    print(f"\n--- {title} ---")
    for i in idx:
        sid = meta.index[i]
        snippet = " ".join(str(text.loc[sid]).split())[:230]
        print(f"[{meta.bioguide.iloc[i]}, {meta.n_words.iloc[i]} words] {snippet}")
show(order[:args.examples], "lowest on PC1")
show(order[::-1][:args.examples], "highest on PC1")
