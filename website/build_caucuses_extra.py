"""Extra 118th-Congress House caucus rosters (Wikipedia oldid snapshots, Sept 2024)
matched to Voteview bioguide IDs.  Run from repo root:  uv run python website/build_caucuses_extra.py

Wikipedia wikitext is fetched on the fly via the MediaWiki API (latest revision before 2024-10-01).
Tuesday Group is the pre-2020 name of the Republican Governance Group (same Wikipedia page);
it is NOT emitted as a separate caucus.
"""
import os, re, time, unicodedata, requests
import pandas as pd

REPO = os.getcwd()
OUT_RAW = os.path.join(REPO, "website", "caucus_rosters_extra_raw.csv")
OUT = os.path.join(REPO, "website", "caucuses_extra_118.csv")
H = {"User-Agent": "stat236-class-project/1.0 (educational)"}
API = "https://en.wikipedia.org/w/api.php"
TS = "2024-10-01T00:00:00Z"

STATES = {"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY","Puerto Rico":"PR","Guam":"GU","American Samoa":"AS","District of Columbia":"DC","Northern Mariana Islands":"MP","U.S. Virgin Islands":"VI","Virgin Islands":"VI"}

# caucus label, wikipedia title, section start, section end
SPECS = [
    ("Republican Study Committee", "Republican Study Committee", "=== Current members ===", "== Notes =="),
    ("Republican Governance Group", "Tuesday Group", "===Current members===", "===Former members==="),
    ("Congressional Black Caucus", "Congressional Black Caucus", "=== United States House of Representatives ===", "== Prominent former members =="),
    ("Congressional Hispanic Caucus", "Congressional Hispanic Caucus", "=== United States House of Representatives ===", "== Controversies =="),
    ("Western Caucus", "Congressional Western Caucus", "== Members ==", "== References =="),
    ("Equality Caucus", "Congressional Equality Caucus", "===Members===", "==Former co-chairs=="),
]

def fetch(title):
    for k in range(6):
        resp = requests.get(API, headers=H, params=dict(action="query", prop="revisions", titles=title, rvlimit=1, rvdir="older", rvstart=TS, rvprop="ids|timestamp|content", rvslots="main", format="json", redirects=1, formatversion=2))
        try: r = resp.json(); break
        except ValueError: time.sleep(3 * (k + 1))  # rate limited
    p = r["query"]["pages"][0]; rev = p["revisions"][0]
    url = f"https://en.wikipedia.org/w/index.php?title={p['title'].replace(' ','_')}&oldid={rev['revid']}"
    print(f"{title}: oldid {rev['revid']} @ {rev['timestamp']}")
    return rev["slots"]["main"]["content"], url

def link_name(s):
    m = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", s)
    return (m.group(2) or m.group(1)).strip()

def st_dist(line, cur_state):
    m = re.search(r"\b([A-Z]{2})\s*[-–]\s*(\d+|AL)\b", line)
    if m:
        d = m.group(2); return m.group(1), ("0" if d == "AL" else str(int(d)))
    m = re.search(r"\(\s*[DR]\s*[–―-]\s*([A-Za-z. ]+?)\s*\)", line)  # (D–North Carolina)
    if m and m.group(1).strip() in STATES: return STATES[m.group(1).strip()], None
    return cur_state, None

rows = []
for label, title, a, b in SPECS:
    txt, url = fetch(title)
    i = txt.index(a); j = txt.index(b, i + len(a))
    cur_state = None
    for line in txt[i:j].splitlines():
        hm = re.match(r"^'''\[\[([^\]|]+)", line.strip())  # state header '''[[Arizona]]'''
        if hm: cur_state = STATES.get(hm.group(1)); continue
        hm = re.match(r"^([A-Z][A-Za-z. ]+):\s*$", line.strip())  # "Arizona:"
        if hm and hm.group(1) in STATES: cur_state = STATES[hm.group(1)]; continue
        if line.lstrip().startswith("*") and "[[" in line:
            st, d = st_dist(line, cur_state)
            rows.append((label, link_name(line), st, d, url))

raw = pd.DataFrame(rows, columns=["caucus","name","state","district","source_url"])
raw["name"] = raw["name"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip()
raw["name"] = raw["name"].replace({"Josh Breechen": "Josh Brecheen"})  # Wikipedia typo

def norm(s):
    s = re.sub(r"[-'’.]", " ", s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z# ]", " ", s)
    return " ".join(w for w in s.split() if w not in {"jr","sr","ii","iii","iv"})

vv = pd.read_csv(os.path.join(REPO, "data/118/voteview/H118_members.csv"))
vv = vv[vv.chamber == "House"].drop_duplicates("bioguide_id")
vv["last"] = vv.bioname.str.split(",").str[0].map(norm)
rest = vv.bioname.str.split(",", n=1).str[1].fillna("")
vv["firsts"] = rest.map(lambda r: set(norm(re.sub(r"[()]", " ", r)).split()))
NICK = {"chuck":"charles","buddy":"earl","dusty":"dustin","jen":"jennifer","mike":"michael","andy":"andrew","greg":"gregory","jim":"james","tom":"thomas","bill":"william","bob":"robert","dan":"daniel","dave":"david","don":"donald","ed":"edward","nick":"nicholas","pete":"peter","chris":"christopher","joe":"joseph","ben":"benjamin","rick":"richard","rich":"richard","raul":"raul","tony":"antonio","hank":"henry","jared":"jared","sheila":"sheila","tim":"timothy","terri":"theresa","debbie":"deborah","gus":"gus"}

def match(r):
    c = vv[vv.state_abbrev == r.state] if r.state else vv
    n = norm(r["name"]); words = n.split()
    def last_ok(L):
        lw = L.split()
        return any(words[i:i+len(lw)] == lw for i in range(len(words)))
    cand = c[c["last"].map(last_ok)]
    if len(cand) == 0:
        cand = c[c["last"].map(lambda L: last_ok(L.split()[0]))]
    if len(cand) > 1 and pd.notna(r.district):
        cand2 = cand[cand.district_code.astype(str) == str(r.district)]
        if len(cand2): cand = cand2
    if len(cand) > 1:
        first = words[0]
        cand2 = cand[cand.firsts.map(lambda f: first in f or NICK.get(first) in f)]
        if len(cand2): cand = cand2
    if len(cand) > 1:
        L = cand["last"].str.len(); cand = cand[L == L.max()]
    return cand.bioguide_id.iloc[0] if len(cand) == 1 else (None if len(cand) == 0 else "AMBIGUOUS")

raw["bioguide"] = raw.apply(match, axis=1)
raw[["caucus","name","state","district","source_url"]].to_csv(OUT_RAW, index=False)
bad = raw[raw.bioguide.isna() | (raw.bioguide == "AMBIGUOUS")]
print("UNMATCHED:\n", bad[["caucus","name","state","district","bioguide"]].to_string())
ok = raw[raw.bioguide.notna() & (raw.bioguide != "AMBIGUOUS")][["bioguide","caucus"]].drop_duplicates()
ok.to_csv(OUT, index=False)
print("raw:\n", raw.caucus.value_counts().to_string())
chk = ok.merge(vv[["bioguide_id","party_code"]], left_on="bioguide", right_on="bioguide_id")
print("matched by party (100=D,200=R):\n", chk.groupby(["caucus","party_code"]).size().to_string())
