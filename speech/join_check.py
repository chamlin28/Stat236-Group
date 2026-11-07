"""Check how well the speech data lines up with the group's member-bill graph.

Usage: python join_check.py 117 [--graph-repo ~/Stat236-Group]
Needs: data/<congress>/edges.csv from the group repo, and crec_<congress>.parquet.
If data/elections/targets.csv exists in the repo, coverage is also reported for the
modeling sample (members with a defined tilt), which is what the models actually use.
Only House speeches are compared, since the graph has House members only.
"""
import argparse, os
import pandas as pd

ELECTION = {115: "2018-11-06", 116: "2020-11-03", 117: "2022-11-08", 118: "2024-11-05"}

ap = argparse.ArgumentParser()
ap.add_argument("congress", type=int)
ap.add_argument("--graph-repo", default=os.path.expanduser("~/Stat236-Group"))
args = ap.parse_args()
c = args.congress

base = os.environ.get("CREC_DATA_DIR", os.path.expanduser("~/crec_data"))
edges = pd.read_csv(os.path.join(args.graph_repo, "data", str(c), "edges.csv"))
members = edges.drop_duplicates("member").set_index("member")["party"]

targets_path = os.path.join(args.graph_repo, "data", "elections", "targets.csv")
model_ids = None
if os.path.exists(targets_path):
    t = pd.read_csv(targets_path)
    model_ids = pd.Index(t[(t.congress == c) & t.tilt.notna()].bioguide.unique())

sp = pd.read_parquet(os.path.join(base, "parsed", f"crec_{c}.parquet"),
                     columns=["date", "chamber", "speaker_bioguide", "speech"])
sp = sp[(sp.chamber == "H") & sp.speaker_bioguide.notna()].copy()
sp["n"] = sp.speech.str.len()
sp["long"] = sp.n >= 500
sp = sp.drop(columns="speech")

def report(df, label):
    g = df.groupby("speaker_bioguide").agg(speeches=("n", "size"), long=("long", "sum"))
    g = g.reindex(members.index).fillna(0)
    print(f"\n--- {label} ---")
    print(f"House speech rows with a member ID: {len(df)}")
    print(f"graph members: {len(g)}")
    for k in (1, 5, 20):
        print(f"  with >= {k:>2} speeches: {(g.speeches >= k).sum():>4} ({(g.speeches >= k).mean():.0%})")
    print(f"  with >= 1 long (500+ char) speech: {(g.long >= 1).sum()} ({(g.long >= 1).mean():.0%})")
    print(f"  speeches per member: median {g.speeches.median():.0f}, mean {g.speeches.mean():.1f}, max {g.speeches.max():.0f}")
    by_party = g.assign(party=members).groupby("party").speeches.agg(["size", "median", lambda s: (s >= 1).mean()])
    by_party.columns = ["members", "median_speeches", "share_with_any"]
    print(by_party.round(2).to_string())
    outside = df[~df.speaker_bioguide.isin(members.index)]
    print(f"rows from speakers NOT in the graph: {len(outside)} ({len(outside)/max(len(df),1):.1%}); "
          f"distinct IDs: {outside.speaker_bioguide.nunique()}")
    print(outside.speaker_bioguide.value_counts().head(5).to_string())

print(f"Congress {c}: dates {sp.date.min().date()} to {sp.date.max().date()}")
report(sp, "all House speeches in the Congress")
if c in ELECTION:
    pre = sp[sp.date < pd.Timestamp(ELECTION[c])]
    report(pre, f"only speeches before Election Day {ELECTION[c]}")

# --- How coverage changes with the minimum speech length (pre-election speeches if known) ---
base_df = sp[sp.date < pd.Timestamp(ELECTION[c])] if c in ELECTION else sp
label = "pre-election" if c in ELECTION else "all dates"
rows = []
for thr in (0, 250, 500, 1000):
    cnt = base_df[base_df.n >= thr].groupby("speaker_bioguide").size()
    g = cnt.reindex(members.index).fillna(0)
    row = {"min_chars": thr, "speech_rows": int((base_df.n >= thr).sum()),
           "graph>=1": int((g >= 1).sum()), "graph>=5": int((g >= 5).sum()),
           "graph>=20": int((g >= 20).sum())}
    if model_ids is not None:
        m = cnt.reindex(model_ids).fillna(0)
        row.update({"model>=1": int((m >= 1).sum()), "model>=5": int((m >= 5).sum()),
                    "model>=20": int((m >= 20).sum()), "model_median": m.median()})
    rows.append(row)
print(f"\n--- coverage by minimum length ({label}) ---")
print(f"graph members: {len(members)}" + (f", modeling sample (defined tilt): {len(model_ids)}" if model_ids is not None else ""))
print(pd.DataFrame(rows).to_string(index=False))
