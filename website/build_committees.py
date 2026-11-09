"""Build 118th House committee memberships (full committees) -> bioguide IDs.

Rosters: Wikipedia "Members, 118th Congress" sections, oldid snapshots from
mid/late 2024 cached in website/wiki_committees/ (re-create with
website/fetch_committees.py, which needs network access to the Wikipedia API).
Inclusion rule: member served on the committee >= 365 days of the Congress
(2023-01-03 .. 2025-01-03), using "(from DATE)" / "(until DATE)" annotations.

Run from repo root:  uv run python website/build_committees.py
"""
import json, os, re, unicodedata
from datetime import date, datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WIKI = os.path.join(HERE, "wiki_committees")
REPO = os.getcwd()
meta = json.load(open(os.path.join(WIKI, "meta.json")))
START, END = date(2023, 1, 3), date(2025, 1, 3)

STATES = {"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY","Puerto Rico":"PR","Guam":"GU","American Samoa":"AS","District of Columbia":"DC","Northern Mariana Islands":"MP","U.S. Virgin Islands":"VI","Virgin Islands":"VI"}

def pdate(s): return datetime.strptime(s.strip(), "%B %d, %Y").date()

rows, dropped = [], []
for com, m in meta.items():
    fn = os.path.join(WIKI, com.replace(" ", "_").replace(",", "").replace("'", "") + ".wiki")
    t = open(fn, encoding="utf-8").read()
    i = t.index("Members, 118th Congress"); j = t.index("\n|}", i)
    for line in t[i:j].splitlines():
        if not line.startswith("*"): continue
        mm = re.match(r"\*\s*\[\[([^\]|]+)(?:\|([^\]]+))?\]\],\s*([A-Za-z. ]+?)\s*(?:,|\(|$)", line.strip())
        name = (mm.group(2) or mm.group(1)).strip(); st = STATES[mm.group(3).strip()]
        s, e = START, END
        f = re.search(r"\(from ([A-Za-z]+ \d+, \d{4})\)", line); u = re.search(r"\(until ([A-Za-z]+ \d+, \d{4})\)", line)
        if f: s = pdate(f.group(1))
        if u: e = pdate(u.group(1))
        days = (e - s).days
        if days < 365:
            dropped.append((com, name, st, days)); continue
        rows.append((com, name, st, m["url"]))
raw = pd.DataFrame(rows, columns=["committee", "name", "state", "source_url"])
raw["name"] = raw["name"].str.replace("�", "'")

def norm(s):
    s = re.sub(r"[-'’.]", " ", s).replace("�", "#")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z# ]", " ", s)
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv"})

vv = pd.read_csv(os.path.join(REPO, "data/118/voteview/H118_members.csv"))
vv = vv[vv.chamber == "House"].drop_duplicates("bioguide_id")
vv["last"] = vv.bioname.str.split(",").str[0].map(norm)
rest = vv.bioname.str.split(",", n=1).str[1].fillna("")
vv["firsts"] = rest.map(lambda r: set(norm(re.sub(r"[()]", " ", r)).split()))
NICK = {"chuck":"charles","buddy":"earl","dusty":"dustin","jen":"jennifer","mike":"michael","andy":"andrew","greg":"gregory","jim":"james","tom":"thomas","bill":"william","bob":"robert","dan":"daniel","dave":"david","don":"donald","ed":"edward","nick":"nicholas","pete":"peter","chris":"christopher","joe":"joseph","ben":"benjamin","rick":"richard","rich":"richard","hal":"harold","mo":"maurice","tim":"timothy","ken":"kenneth","steve":"steven","jimmy":"james","jerry":"gerald","sam":"samuel","liz":"elizabeth","sheila":"sheila","jackie":"jacqueline","mark":"mark","ro":"rohit","hank":"henry","gus":"gus","bruce":"bruce","terri":"theresa","debbie":"deborah","raul":"raul","dina":"dina","buck":"buck"}

def match(r):
    c = vv[vv.state_abbrev == r.state]
    words = norm(r["name"]).split()
    def last_ok(L):
        lw = L.split(); pat = [re.compile(x.replace("#", "[a-z]") + "$") for x in lw]
        return any(len(words[i:i+len(lw)]) == len(lw) and all(p.match(w) for p, w in zip(pat, words[i:i+len(lw)])) for i in range(len(words)))
    cand = c[c["last"].map(last_ok)]
    if len(cand) == 0:
        cand = c[c["last"].map(lambda L: last_ok(L.split()[0]))]
    if len(cand) > 1:
        first = words[0]
        c2 = cand[cand.firsts.map(lambda f: first in f or NICK.get(first) in f)]
        if len(c2): cand = c2
    if len(cand) > 1:
        L = cand["last"].str.len(); cand = cand[L == L.max()]
    return cand.bioguide_id.iloc[0] if len(cand) == 1 else (None if len(cand) == 0 else "AMBIGUOUS")

raw["bioguide"] = raw.apply(match, axis=1)
raw[["committee", "name", "state", "source_url"]].to_csv(os.path.join(REPO, "website", "committee_rosters_raw.csv"), index=False)
bad = raw[raw.bioguide.isna() | (raw.bioguide == "AMBIGUOUS")]
print("UNMATCHED/AMBIGUOUS:\n", bad[["committee", "name", "state", "bioguide"]].to_string())
ok = raw[raw.bioguide.notna() & (raw.bioguide != "AMBIGUOUS")][["bioguide", "committee"]].drop_duplicates()
ok.to_csv(os.path.join(REPO, "website", "committees_118.csv"), index=False)
print(ok.committee.value_counts().to_string())
print("dropped (<365 days):", dropped)
chk = raw.merge(vv[["bioguide_id", "bioname"]], left_on="bioguide", right_on="bioguide_id", how="left")
chk[["committee", "name", "state", "bioname"]].to_csv(os.path.join(os.environ.get("TEMP", "."), "committee_match_check.csv"), index=False)
print("dups:", raw.duplicated(["committee", "bioguide"]).sum(), "members:", ok.bioguide.nunique())
