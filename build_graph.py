"""Build the House member <-> bill vote graph for a given Congress.

Nodes : House members, and every bill/joint resolution with a recorded House final-passage roll call.
Edges : member voted Yea on the bill's final House passage vote.
Bill nodes carry an `outcome`: enacted | passed_house_not_enacted | failed, plus yea/nay counts.
Sponsors are NOT in the graph; they are saved separately in sponsors.csv.

Data sources
  Voteview bulk CSVs (no key)         -> members, roll calls, votes
  Congress.gov API (CONGRESS_API_KEY) -> which bills became law, bill sponsors.
    Responses are cached in data/raw/. Without a key, only cached responses are used and
    outcomes fall back to passed_house_not_enacted / failed (no 'enacted'), and sponsors are skipped.

  Bill text -> data/<congress>/text/<bill>.txt (skip with --no-text). With a key, the version is chosen from
    the Congress.gov text-versions list (enrolled for laws, otherwise the version the House voted on);
    without one, GovInfo is tried with the standard version codes.

The API key is read from the CONGRESS_API_KEY environment variable or a line CONGRESS_API_KEY=... in .env

Output in data/<congress>/: graph.graphml, bills.csv, edges.csv, sponsors.csv, text/

Run:  uv run python build_graph.py --congress 118
"""
import argparse
import hashlib
import html
import json
import os
import re
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import requests

API = "https://api.congress.gov/v3"
VOTEVIEW = "https://voteview.com/static/data/out/{kind}/H{congress}_{kind}.csv"
RAW = Path("data/raw")
# vote_question values meaning "the bill itself was passed/agreed to" (not amendments or procedure)
PASSAGE = r"On Passage|and Pass|Concur in the Senate Amendment|Conference Report|^Passage"
PARTY = {100: "D", 200: "R"}
# Text version to prefer, in order, matched against the Congress.gov version type (key set)
VERSION_PREFS = {"enacted": ["Enrolled"]}
VERSION_PREFS_DEFAULT = ["Engrossed in House", "Engrossed Amendment House", "Engrossed in Senate",
                         "Engrossed Amendment Senate", "Reported in House", "Reported in Senate",
                         "Introduced in House", "Introduced in Senate"]
# GovInfo version codes to try per outcome (no key)
GOVINFO = "https://www.govinfo.gov/content/pkg/BILLS-{c}{t}{n}{code}/html/BILLS-{c}{t}{n}{code}.htm"
GOVINFO_CODES = {"enacted": ["enr"], "passed_house_not_enacted": ["eh", "es", "eas", "eah", "enr", "rh", "ih", "is"],
                 "failed": ["rh", "ih", "rs", "is"]}


def get_api_key():
    """CONGRESS_API_KEY from the environment, else from a .env file in the working directory."""
    key = os.environ.get("CONGRESS_API_KEY")
    if not key and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.strip().startswith("CONGRESS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    return key


def api_get(path, **params):
    """Cached Congress.gov GET. Returns parsed JSON, or None if not cached and no API key is set."""
    query = dict(sorted(params.items()))
    tag = hashlib.md5(f"{path}?{query}".encode()).hexdigest()[:10]
    cache = RAW / f"{path.strip('/').replace('/', '_')}_{tag}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    key = get_api_key()
    if not key:
        return None
    for attempt in range(5):  # the API occasionally times out or rate-limits; back off and retry
        try:
            r = requests.get(f"{API}{path}", params={**query, "api_key": key, "format": "json"}, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            break
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
            if attempt == 4:
                raise
            time.sleep(5 * 2 ** attempt)
    r.raise_for_status()
    RAW.mkdir(parents=True, exist_ok=True)
    cache.write_text(r.text)
    return r.json()


def fetch_enacted(congress):
    """{'HR3746': '118-5', ...} from the list of public laws, or None if unavailable."""
    enacted, offset = {}, 0
    while True:
        page = api_get(f"/law/{congress}/pub", limit=250, offset=offset)
        if page is None:
            return None
        for b in page["bills"]:
            law = (b.get("laws") or [{}])[0].get("number")
            enacted[f"{b['type'].upper()}{b['number']}"] = law
        offset += 250
        if not page.get("pagination", {}).get("next") or not page["bills"]:
            return enacted


def fetch_sponsors(congress, bill_numbers):
    """Primary sponsor of each bill -> DataFrame (empty if API unavailable)."""
    rows = []
    for bn in bill_numbers:
        btype, num = re.match(r"([A-Z]+)(\d+)$", bn).groups()
        data = api_get(f"/bill/{congress}/{btype.lower()}/{num}")
        if data is None:  # not cached and no API key: skip this bill
            continue
        s = (data["bill"].get("sponsors") or [{}])[0]
        rows.append({"bill": bn, "bioguide": s.get("bioguideId"), "name": s.get("fullName"),
                     "party": s.get("party"), "state": s.get("state"), "district": s.get("district")})
    return pd.DataFrame(rows)


def clean_text(raw):
    """GovInfo/Congress.gov text pages are <pre> blocks (or XML); strip tags and entities."""
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip() + "\n"


def pick_version(versions, outcome):
    """Choose one entry of the Congress.gov textVersions list."""
    for pref in VERSION_PREFS.get(outcome, VERSION_PREFS_DEFAULT):
        for v in versions:
            if pref.lower() in (v.get("type") or "").lower():
                return v
    return max(versions, key=lambda v: v.get("date") or "", default=None)  # fall back to the latest


def fetch_bill_text(congress, bill, outcome, folder, session):
    """Save the bill text to folder/<bill>.txt. Returns the version label, or '' if none found."""
    out = folder / f"{bill}.txt"
    if out.exists():
        return out.read_text(errors="ignore").split("\n", 1)[0].strip("[] ")  # first line = version label
    btype, num = re.match(r"([A-Z]+)(\d+)$", bill).groups()
    candidates = []  # (label, url)
    data = api_get(f"/bill/{congress}/{btype.lower()}/{num}/text")
    if data is not None:
        v = pick_version(data.get("textVersions", []), outcome)
        fmts = {f["type"]: f["url"] for f in (v or {}).get("formats", [])}
        for kind in ["Formatted Text", "Formatted XML"]:
            if kind in fmts:
                candidates.append((v["type"], fmts[kind]))
                break
    else:
        for code in GOVINFO_CODES[outcome]:
            candidates.append((code.upper(), GOVINFO.format(c=congress, t=btype.lower(), n=num, code=code)))
    for label, url in candidates:
        r = session.get(url, timeout=60)
        # GovInfo answers a missing version with HTTP 200 and a "Page Not Found" page
        if r.status_code == 200 and r.text.strip() and not re.search(r"<title>[^<]*Page Not Found", r.text):
            out.write_text(f"[{label}]\n" + clean_text(r.text), encoding="utf-8")
            return label
    return ""


def load_voteview(congress, folder):
    """Download (once) and read the three Voteview House CSVs."""
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind in ["members", "rollcalls", "votes"]:
        f = folder / f"H{congress}_{kind}.csv"
        if not f.exists():
            print(f"      downloading {f.name}")
            r = requests.get(VOTEVIEW.format(kind=kind, congress=congress), timeout=120)
            r.raise_for_status()
            f.write_bytes(r.content)
        out[kind] = pd.read_csv(f)
    return out["members"], out["rollcalls"], out["votes"]


def main(congress, get_text=True):
    out_dir = Path("data") / str(congress)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Voteview data for the {congress}th Congress ...")
    members, rolls, votes = load_voteview(congress, out_dir / "voteview")
    members = members[(members["chamber"] == "House") & members["bioguide_id"].notna()].copy()
    members["party"] = members["party_code"].map(PARTY).fillna("Other")
    print(f"      {len(members)} House members, {len(rolls)} roll calls")

    print("[2/6] final passage vote per bill ...")
    passage = rolls[rolls["vote_question"].str.contains(PASSAGE, na=False) & rolls["bill_number"].notna()]
    bills = passage.sort_values("rollnumber").groupby("bill_number").tail(1).copy()

    print("[3/6] which bills became law (Congress.gov) ...")
    enacted = fetch_enacted(congress)
    if enacted is None:
        print("      no API key and nothing cached: outcome 'enacted' unavailable")
        enacted = {}
    bills["law"] = bills["bill_number"].map(enacted)
    bills["outcome"] = bills["vote_result"].map({"Passed": "passed_house_not_enacted"}).fillna("failed")
    bills.loc[bills["law"].notna(), "outcome"] = "enacted"
    print("      " + ", ".join(f"{k}: {v}" for k, v in bills["outcome"].value_counts().items()))

    print("[4/6] sponsors (stored, not in graph) ...")
    sponsors = fetch_sponsors(congress, bills["bill_number"])
    if sponsors.empty:
        print("      skipped (no API key and not cached)")
    else:
        sponsors.to_csv(out_dir / "sponsors.csv", index=False)
        print(f"      sponsors for {len(sponsors)} of {len(bills)} bills saved"
              + ("" if len(sponsors) == len(bills) else " (set CONGRESS_API_KEY to fetch the rest)"))

    print("[5/6] bill text ...")
    bills["text_version"] = ""
    if get_text:
        text_dir = out_dir / "text"
        text_dir.mkdir(exist_ok=True)
        with requests.Session() as session:
            bills["text_version"] = [fetch_bill_text(congress, b, o, text_dir, session)
                                     for b, o in zip(bills["bill_number"], bills["outcome"])]
        n = (bills["text_version"] != "").sum()
        print(f"      text saved for {n} of {len(bills)} bills -> {text_dir}/")
    else:
        print("      skipped (--no-text)")

    print("[6/6] building graph ...")
    yeas = votes[votes["cast_code"].isin([1, 2, 3])].merge(bills[["rollnumber", "bill_number"]], on="rollnumber")
    yeas = yeas.merge(members[["icpsr", "bioguide_id", "party"]], on="icpsr")
    edges = yeas.rename(columns={"bioguide_id": "member"})[["member", "bill_number", "rollnumber", "party"]]
    edges = edges.rename(columns={"bill_number": "bill"})

    G = nx.Graph()
    for m in members.itertuples():
        G.add_node(m.bioguide_id, kind="member", name=m.bioname, party=m.party, state=m.state_abbrev,
                   district=int(m.district_code))
    for b in bills.itertuples():
        G.add_node(b.bill_number, kind="bill", outcome=b.outcome, law=str(b.law) if pd.notna(b.law) else "",
                   yea_count=int(b.yea_count), nay_count=int(b.nay_count), rollnumber=int(b.rollnumber),
                   date=b.date, question=b.vote_question, text_version=b.text_version)
    G.add_edges_from((e.member, e.bill) for e in edges.itertuples() if e.member in G)

    nx.write_graphml(G, out_dir / "graph.graphml")
    bills.drop(columns=["nominate_mid_1", "nominate_mid_2", "nominate_spread_1", "nominate_spread_2",
                        "nominate_log_likelihood"], errors="ignore").to_csv(out_dir / "bills.csv", index=False)
    edges.to_csv(out_dir / "edges.csv", index=False)
    print(f"      {G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--congress", type=int, default=118, help="Congress number, e.g. 117 or 118")
    ap.add_argument("--no-text", action="store_true", help="skip downloading bill text")
    args = ap.parse_args()
    main(args.congress, get_text=not args.no_text)
