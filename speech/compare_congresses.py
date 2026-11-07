"""Compare the parsed Congresses side by side, using lengths so it stays light on memory.

Usage: python compare_congresses.py
Reads every crec_*.parquet in $CREC_DATA_DIR/parsed (default ~/crec_data/parsed).
"""
import glob, os
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

base = os.environ.get("CREC_DATA_DIR", os.path.expanduser("~/crec_data"))
files = sorted(glob.glob(os.path.join(base, "parsed", "crec_*.parquet")))
if not files:
    raise SystemExit("No parquet files found in " + base + "/parsed")

frames = []
for f in files:
    cong = int(os.path.basename(f).split("_")[1].split(".")[0])
    t = pq.read_table(f, columns=["date", "chamber", "speaker_bioguide", "speech"])
    frames.append(pd.DataFrame({
        "congress": cong,
        "date": t["date"].to_pandas(),
        "chamber": t["chamber"].to_pandas(),
        "no_id": pc.is_null(t["speaker_bioguide"]).to_pandas(),
        "n": pc.utf8_length(t["speech"]).to_pandas(),
    }))
    del t
d = pd.concat(frames, ignore_index=True)
d["long"] = d.n >= 500          # substantive remarks, not one-line procedural turns
d["year"] = d.date.dt.year

def summarize(g):
    days = g.date.nunique()
    return pd.Series({
        "days": days,
        "rows": len(g),
        "rows_per_day": round(len(g) / days, 1),
        "long_rows": int(g.long.sum()),
        "long_per_day": round(g.long.sum() / days, 1),
        "median_chars": int(g.n.median()),
        "empty_rows": int((g.n == 0).sum()),
        "pct_no_id": round(100 * g.no_id.mean(), 1),
    })

pd.set_option("display.width", 160)
print("BY CONGRESS (long = 500+ characters)")
print(d.groupby("congress").apply(summarize, include_groups=False).to_string())
print("\nBY CALENDAR YEAR")
print(d.groupby("year").apply(summarize, include_groups=False).to_string())
print("\nBY CONGRESS AND CHAMBER")
print(d.groupby(["congress", "chamber"]).apply(summarize, include_groups=False).to_string())
