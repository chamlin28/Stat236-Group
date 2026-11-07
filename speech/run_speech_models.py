"""One command: build the speech features, then compare models with and without them.

Usage:  python run_speech_models.py
Options: --skip-features (reuse an existing speech_features.csv), --train 115 116 --test 118

Protocol (same as the group's tilt_features notebook): train on the TRAIN Congresses, choose the ridge
penalty by leave-one-Congress-out inside TRAIN, evaluate on the TEST Congress. Features are z-scored
within each Congress and missing values filled with 0. Out-of-sample R2 is relative to the training mean.
Because there is only one test year, a paired bootstrap over test members gives an interval for the
improvement in mean squared error over the baseline (party + previous share). Graph features are not
included here (the notebook found they add nothing reliable); only the speech features are tested.
"""
import argparse, os, subprocess, sys
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--train", type=int, nargs="+", default=[115, 116])
ap.add_argument("--test", type=int, default=118)
ap.add_argument("--targets", default=os.path.expanduser("~/Stat236-Group/data/elections/targets.csv"))
ap.add_argument("--skip-features", action="store_true")
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()
TRAIN, TEST = args.train, args.test
base = os.environ.get("CREC_DATA_DIR", os.path.expanduser("~/crec_data"))
emb = os.path.join(base, "embeddings")
sf_path = os.path.join(emb, "speech_features.csv")
here = os.path.dirname(os.path.abspath(__file__))

# ---- 1. Build the speech features ----
if not args.skip_features or not os.path.exists(sf_path):
    print("=" * 70, "\nSTEP 1: speech features\n" + "=" * 70, flush=True)
    cmd = [sys.executable, os.path.join(here, "speech_features.py"),
           "--congresses", *map(str, TRAIN + [TEST]), "--train", *map(str, TRAIN),
           "--targets", args.targets]
    subprocess.run(cmd, check=True)

# ---- 2. Assemble the modeling table (as in notebook cell 5) ----
print("\n" + "=" * 70 + "\nSTEP 2: models\n" + "=" * 70, flush=True)
targets = pd.read_csv(args.targets)
sf = pd.read_csv(sf_path)
SPEECH = [c for c in sf.columns if c not in ("congress", "bioguide")]
d = targets[targets.congress.isin(TRAIN + [TEST])].merge(sf, on=["congress", "bioguide"], how="left")
d = d[d.tilt.notna() & d.party.isin(["D", "R"])].copy()
d["tilt_c"] = d.tilt - d.groupby(["congress", "party"]).tilt.transform("mean")
d["is_R"] = (d.party == "R").astype(float)
d["prev_share"] = d.prev_two_party_share
z = lambda x: (x - x.mean()) / x.std()
d[SPEECH + ["prev_share"]] = d.groupby("congress")[SPEECH + ["prev_share"]].transform(z)
d[SPEECH] = d[SPEECH].fillna(0)
print("Rows by Congress and party:")
print(d.groupby(["congress", "party"]).size().unstack().to_string())
print("Share of rows with an overall speech embedding:",
      d.assign(has=sf.set_index(["congress", "bioguide"]).all_pc1.reindex(
          pd.MultiIndex.from_frame(d[["congress", "bioguide"]])).notna().values)
       .groupby("congress").has.mean().round(2).to_dict())

# ---- 3. Models ----
def ridge(X, y, lam):
    mu, ym = X.mean(0), y.mean()
    beta = np.linalg.solve((X - mu).T @ (X - mu) + lam * np.eye(X.shape[1]), (X - mu).T @ (y - ym))
    return (lambda Xn: ym + (Xn - mu) @ beta), beta

def run(cols, target):
    tr, te = d[d.congress.isin(TRAIN)], d[d.congress == TEST]
    cv = {}
    for lam in [1, 10, 100, 1000, 10000]:
        cv[lam] = sum(((b[target].values - ridge(a[cols].values, a[target].values, lam)[0](b[cols].values)) ** 2).sum()
                      for a, b in [(tr[tr.congress != h], tr[tr.congress == h]) for h in TRAIN])
    lam = min(cv, key=cv.get)
    pred, _ = ridge(tr[cols].values, tr[target].values, lam)
    p, y = pred(te[cols].values), te[target].values
    sse0 = ((y - tr[target].mean()) ** 2).sum()
    return {"lambda": lam, "oos_R2": 1 - ((y - p) ** 2).sum() / sse0,
            "test_corr": float(np.corrcoef(p, y)[0, 1]), "RMSE": float(np.sqrt(((y - p) ** 2).mean()))}, p, y

basic = ["log_speeches", "log_words", "words_per_speech", "share_extension", "log_floor", "log_ext"]
pa = lambda s, k: [f"{s}_pc{j}_pa" for j in range(1, k + 1)]
base_cols = ["is_R", "prev_share"]
sets = {
    "party only": ["is_R"],
    "BASELINE: party + prev share": base_cols,
    "+ speech basics": base_cols + basic,
    "+ overall PCs 1-3 (party-adj)": base_cols + pa("all", 3),
    "+ overall PCs 1-5 (party-adj)": base_cols + pa("all", 5),
    "+ floor PCs 1-3 + extension PCs 1-3": base_cols + pa("floor", 3) + pa("ext", 3),
    "+ basics + overall PCs 1-3": base_cols + basic + pa("all", 3),
    "+ everything speech": base_cols + basic + [c for c in SPEECH if c.endswith("_pa")],
    "speech only (+party): basics + PCs 1-3": ["is_R"] + basic + pa("all", 3),
}
rng = np.random.default_rng(236)
rows = []
for target in ["tilt_c", "tilt"]:
    results = {name: run(cols, target) for name, cols in sets.items()}
    _, pb, y = results["BASELINE: party + prev share"]
    n = len(y); draws = rng.integers(0, n, size=(args.boot, n))
    for name, (m, p, _) in results.items():
        gain = ((y - pb) ** 2) - ((y - p) ** 2)          # positive = better than baseline, per member
        boot = gain[draws].mean(1)
        rows.append({"target": target, "features": name, **{k: round(v, 3) if k != "lambda" else v for k, v in m.items()},
                     "MSE_gain_vs_baseline": round(float(gain.mean()), 5),
                     "gain_95%_CI": f"[{np.percentile(boot, 2.5):.5f}, {np.percentile(boot, 97.5):.5f}]"})
res = pd.DataFrame(rows)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)
for target in ["tilt_c", "tilt"]:
    print(f"\nTarget: {target}  (train {TRAIN}, test {TEST}, n_test = {int((d.congress == TEST).sum())})")
    print(res[res.target == target].drop(columns="target").to_string(index=False))
out = os.path.join(emb, "speech_model_results.csv")
res.to_csv(out, index=False)
print(f"\nSaved {out}")
print("Reading guide: 'MSE_gain_vs_baseline' > 0 means better than party + previous share. If the 95% interval "
      "includes 0, the improvement is not distinguishable from noise with one test year.")
