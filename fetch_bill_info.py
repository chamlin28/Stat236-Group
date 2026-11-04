"""Fetch title, policy area and latest CRS summary for every bill in data/<congress>/bills.csv.

Writes data/<congress>/bill_info.csv (bill, title, policy_area, summary). Uses the cached Congress.gov helper
from build_graph.py, so it needs CONGRESS_API_KEY (env or .env) for anything not already in data/raw.

    uv run python fetch_bill_info.py --congress 118
"""
import argparse
import html
import re
from pathlib import Path

import pandas as pd

from build_graph import api_get


def strip_html(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def main(congress):
    out_dir = Path("data") / str(congress)
    bills = pd.read_csv(out_dir / "bills.csv")["bill_number"]
    rows = []
    for bn in bills:
        btype, num = re.match(r"([A-Z]+)(\d+)$", bn).groups()
        path = f"/bill/{congress}/{btype.lower()}/{num}"
        rec = (api_get(path) or {}).get("bill", {})
        sums = (api_get(f"{path}/summaries") or {}).get("summaries", [])
        latest = max(sums, key=lambda s: s.get("updateDate", ""), default={})   # most recent CRS version
        rows.append({"bill": bn, "title": rec.get("title", ""), "policy_area": (rec.get("policyArea") or {}).get("name", ""),
                     "summary": strip_html(latest.get("text"))})
    info = pd.DataFrame(rows)
    info.to_csv(out_dir / "bill_info.csv", index=False)
    print(f"{congress}: {len(info)} bills, {(info.title != '').sum()} titles, {(info.summary != '').sum()} summaries, "
          f"{(info.policy_area != '').sum()} policy areas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--congress", type=int, nargs="+", default=[118])
    for c in ap.parse_args().congress:
        main(c)
