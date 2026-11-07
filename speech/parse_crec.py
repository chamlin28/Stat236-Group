"""Parse downloaded CREC JSON for one Congress into a pandas DataFrame (HDF5).

Usage: python parse_crec.py 118 [--hdf]

Differences from ParseCREC.ipynb: the stray debug line is removed, speeches
that do not start with the speaker name are logged and skipped instead of
halting the run, and only rows with the requested Congress number are kept.
"""
import datetime, json, logging, os, sys, time
from collections import OrderedDict
from xml.etree import ElementTree as ET
import pandas as pd
from config import RAW_DIR, OUT_DIR, date_range

congress = int(sys.argv[1])
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(message)s")
os.makedirs(OUT_DIR, exist_ok=True)

MODS = "{http://www.loc.gov/mods/v3}"

def squeeze(s):
    return " ".join(s.split())

start, end = date_range(congress)
day = datetime.datetime.strptime(start, "%Y-%m-%d")
end_day = datetime.datetime.strptime(end, "%Y-%m-%d")

data, skipped = [], {"no_prefix": 0, "clerk": 0, "bad_file": 0}
days_parsed = 0

while day <= end_day:
    ds = day.strftime("%Y-%m-%d")
    base = os.path.join(RAW_DIR, str(day.year), "CREC-" + ds)
    jdir = os.path.join(base, "json")
    day += datetime.timedelta(days=1)

    if not os.path.exists(os.path.join(jdir, "JSON_PARSING_DONE")):
        continue
    try:
        root = ET.parse(os.path.join(base, "mods.xml")).getroot()
        congress_num = int(root.find("./%sextension/%scongress" % (MODS, MODS)).text)
    except Exception as e:
        logging.warning("%s: cannot read Congress number (%s), skipping day", ds, e)
        continue
    if congress_num != congress:
        continue

    days_parsed += 1
    for fname in sorted(os.listdir(jdir)):
        if not fname.endswith(".json"):
            continue
        try:
            o = json.load(open(os.path.join(jdir, fname)))
            h = o["header"]
            chamber = {"House": "H", "Senate": "S"}.get(h["chamber"])
            if chamber is None:
                skipped["bad_file"] += 1
                continue
        except Exception as e:
            logging.warning("%s/%s unreadable (%s)", ds, fname, e)
            skipped["bad_file"] += 1
            continue

        for ndx, c in enumerate(o.get("content", [])):
            if c.get("kind") != "speech":
                continue
            speaker = (c.get("speaker") or "").strip()
            text = (c.get("text") or "").strip()
            if "clerk" in speaker.lower():
                skipped["clerk"] += 1
                continue
            for prefix in (speaker + ".", speaker + " ."):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            else:
                skipped["no_prefix"] += 1
                logging.info("No speaker prefix: %s | %s", speaker, text[:80])
                continue
            data.append(OrderedDict(
                speech_id=f"{o['id']}-{ndx}",
                speech=squeeze(text),
                chamber=chamber,
                is_extension=h.get("extension"),
                date=ds,
                speaker=speaker,
                speaker_bioguide=c.get("speaker_bioguide"),
                vol=h.get("vol"),
                num=h.get("num"),
                congress_num=congress_num,
                pages=h.get("pages"),
                doc_title=o.get("doc_title"),
                title=o.get("title"),
            ))
    if days_parsed % 25 == 0:
        logging.info("%d days parsed, %d speeches so far", days_parsed, len(data))

logging.info("Days parsed: %d, speeches: %d, skipped: %s",
             days_parsed, len(data), skipped)
if not data:
    sys.exit("No data found. Check RAW_DIR and that the download finished.")

df = pd.DataFrame(data)
df["date"] = pd.to_datetime(df["date"])
for col in ("vol", "num"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.astype({"speech_id": str, "speech": str, "chamber": str,
                "speaker": str, "speaker_bioguide": str, "pages": str,
                "doc_title": str, "title": str})
df = df.set_index("speech_id")

# Parquet is the main output (readable in R with arrow::read_parquet).
out = os.path.join(OUT_DIR, f"crec_{congress}.parquet")
df.to_parquet(out, index=True)
logging.info("Saved %s", out)

# Optional HDF5 output for Python-only use: python parse_crec.py 118 --hdf
if "--hdf" in sys.argv:
    out_h = os.path.join(OUT_DIR, f"crec_{congress}.hdf")
    df.to_hdf(out_h, key="RecentCRECData", format="fixed", mode="w", complevel=5)
    logging.info("Saved %s", out_h)
missing = df.speaker_bioguide.isna() | df.speaker_bioguide.isin(["", "None", "nan"])
print("Share of rows with missing bioguide:", round(missing.mean(), 4))
print("Most common speakers with missing bioguide:")
print(df.loc[missing, "speaker"].value_counts().head(10).to_string())
