"""Early check on raw downloaded JSON: do speech items carry member IDs?

Usage: python check_ids.py 2011 [max_files]
Reads up to max_files (default 300) JSON files from that year while a job is running.
"""
import glob, json, os, sys
from collections import Counter

year = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 300
base = os.environ.get("CREC_DATA_DIR", os.path.expanduser("~/crec_data"))
files = sorted(glob.glob(os.path.join(base, "raw", "output", year, "CREC-*", "json", "*.json")))
print(f"{len(files)} JSON files found for {year}; reading up to {limit}")

speeches = with_id = 0
missing_named = Counter()
for f in files[:limit]:
    try:
        items = json.load(open(f)).get("content", [])
    except Exception as e:
        print("unreadable:", f, e)
        continue
    for c in items:
        if c.get("kind") != "speech":
            continue
        speeches += 1
        if c.get("speaker_bioguide"):
            with_id += 1
        elif not (c.get("speaker") or "").startswith("The "):
            missing_named[c.get("speaker")] += 1

print("speech items:", speeches)
if speeches:
    print("with member ID: %.1f%%" % (100 * with_id / speeches))
    print("named members without ID:", sum(missing_named.values()))
    print(missing_named.most_common(8))
