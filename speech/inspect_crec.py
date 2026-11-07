"""Print a quick summary of a parsed Congress file.

Usage: python inspect_crec.py 117
       python inspect_crec.py --file /path/to/crec_117.parquet
"""
import argparse, os, sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("congress", nargs="?", type=int)
ap.add_argument("--file")
args = ap.parse_args()

if args.file:
    path = args.file
elif args.congress:
    base = os.environ.get("CREC_DATA_DIR", os.path.expanduser("~/crec_data"))
    path = os.path.join(base, "parsed", f"crec_{args.congress}.parquet")
else:
    sys.exit("Give a Congress number or --file")

pd.set_option("display.width", 140)
pd.set_option("display.max_colwidth", 60)

d = pd.read_parquet(path)
print("FILE:", path)
print("ROWS (speeches):", len(d))
print("\nCOLUMNS:")
print(d.dtypes.to_string())

print("\nDATE RANGE:", d.date.min().date(), "to", d.date.max().date())
print("DAYS WITH SPEECHES:", d.date.nunique())
print("CONGRESS NUMBERS PRESENT:", sorted(d.congress_num.unique().tolist()))

print("\nBY CHAMBER:")
print(d.chamber.value_counts().to_string())

print("\nSPEECHES PER MONTH:")
print(d.groupby(d.date.dt.to_period("M")).size().to_string())

dates = pd.Series(sorted(d.date.unique()))
gaps = dates.diff().dt.days
big = pd.DataFrame({"after": dates.shift(1)[gaps > 30], "resumes": dates[gaps > 30],
                    "gap_days": gaps[gaps > 30]})
print("\nGAPS LONGER THAN 30 DAYS (recesses are normal, long unexplained gaps are not):")
print(big.to_string(index=False) if len(big) else "none")

L = d.speech.str.len()
print("\nSPEECH LENGTH (characters):")
print(L.describe().round(0).to_string())

miss = d.speaker_bioguide.isna()
print("\nMISSING BIOGUIDE: %.1f%% of rows" % (100 * miss.mean()))
print("Named members (not 'The ...') missing a bioguide:",
      int((miss & ~d.speaker.str.startswith("The ")).sum()))
print("DISTINCT BIOGUIDE IDS (members):", d.speaker_bioguide.nunique())

print("\nTOP 10 SPEAKERS (members with an ID):")
print(d[~miss].groupby(["speaker_bioguide", "speaker"]).size()
      .sort_values(ascending=False).head(10).to_string())

ex = d[L.between(400, 1500)].sample(1, random_state=1).iloc[0]
print("\nEXAMPLE ROW:")
print({k: ex[k] for k in ["date", "chamber", "speaker", "speaker_bioguide", "title"]})
print(ex.speech[:600])
