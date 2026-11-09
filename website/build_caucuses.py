"""Parse 118th-Congress caucus rosters (Wikipedia oldid snapshots + archived
Main Street Caucus page) and match them to Voteview bioguide IDs.

Run from repo root:  uv run python <this file>
"""
import csv, html, json, os, re, sys, unicodedata
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WIKI = os.path.join(HERE, "wiki")
REPO = os.getcwd()
OUT_RAW = os.path.join(REPO, "website", "caucus_rosters_raw.csv")
OUT = os.path.join(REPO, "website", "caucuses_118.csv")
meta = json.load(open(os.path.join(WIKI, "meta.json")))

STATES = {"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY","Puerto Rico":"PR","Guam":"GU","American Samoa":"AS","District of Columbia":"DC","Northern Mariana Islands":"MP","U.S. Virgin Islands":"VI","Virgin Islands":"VI"}

def read(fn):
    return open(os.path.join(WIKI, fn), encoding="utf-8").read()

def section(txt, start, end):
    i = txt.index(start); j = txt.index(end, i)
    return txt[i:j]

def link_name(s):
    m = re.search(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", s)
    return (m.group(2) or m.group(1)).strip()

def dist(code):  # "CA-04" / "ND-AL" / "NE–02" -> ("CA", "4")
    m = re.search(r"\b([A-Z]{2})\s*[-–]\s*(\d+|AL)\b", code)
    if not m: return None, None
    d = m.group(2)
    return m.group(1), ("0" if d == "AL" else str(int(d)))

rows = []  # caucus, name, state, district, source_url

# Freedom Caucus: current list + members who left / were removed during the 118th
t = read("Freedom_Caucus.wiki"); url = meta["Freedom Caucus"]["url"]
for line in section(t, "=== Current members ===", "=== Former members ===").splitlines():
    if line.lstrip().startswith("*"):
        st, d = dist(line); rows.append(("Freedom Caucus", link_name(line), st, d, url))
for line in section(t, "====Left caucus====", "====Lost renomination").splitlines():
    if line.lstrip().startswith("*") and re.search(r"\((?:left|removed|voted out)[^)]*\b(2023|2024)\)", line):
        name = link_name(line)
        if name == "David Schweikert":  # quit Jan 2023, at the very start of the 118th
            continue
        st = STATES[re.search(r"\]\] of ([A-Za-z ]+?) \(", line).group(1)]
        rows.append(("Freedom Caucus", name, st, None, url))

# Blue Dogs
t = read("Blue_Dogs.wiki"); url = meta["Blue Dogs"]["url"]
for line in section(t, "== Current members ==", "== See also ==").splitlines():
    if line.startswith("*"):
        st, d = dist(line); rows.append(("Blue Dogs", link_name(line), st, d, url))

# New Democrats
t = read("New_Democrats.wiki"); url = meta["New Democrats"]["url"]
for line in section(t, "== Membership ==", "== Campaign arm ==").splitlines():
    if line.startswith("*"):
        st, d = dist(line); rows.append(("New Democrats", link_name(line), st, d, url))

# Problem Solvers
t = read("Problem_Solvers.wiki"); url = meta["Problem Solvers"]["url"]
for line in section(t, "=== Democrats ===", "=== Former members ===").splitlines():
    if line.startswith("*"):
        m = re.search(r"\]\] of \[\[([^\]|]+)", line)
        rows.append(("Problem Solvers", link_name(line), STATES[m.group(1).replace(" (state)", "").replace("Georgia (U.S. state)", "Georgia")], None, url))

# Progressive Caucus (House table only)
t = read("Progressive_Caucus.wiki"); url = meta["Progressive Caucus"]["url"]
cur = (None, None)
for line in section(t, "=== House members ===", "== Affiliate organizations ==").splitlines():
    m = re.match(r"\|\s*\[\[[^\]|]*congressional district[^\]|]*\|([^\]]+)\]\]", line)
    if m:
        cur = dist(m.group(1))
        if cur[0] is None:  # e.g. at-large written differently
            cur = (None, None)
    m = re.search(r"\{\{sortname\|([^|}]+)\|([^|}]+)", line)
    if m:
        rows.append(("Progressive Caucus", f"{m.group(1)} {m.group(2)}", cur[0], cur[1], url))
        cur = (None, None)

# Main Street Caucus: archived official roster (Wayback, 2024-09-18)
url = "http://web.archive.org/web/20240918175936/https://mainstreetcaucus.house.gov/membership"
h = read("mainstreet_wayback.html")
h = re.sub(r"<script.*?</script>|<style.*?</style>", "", h, flags=re.S)
lines = [l.strip() for l in html.unescape(re.sub(r"<[^>]+>", "\n", h)).split("\n") if l.strip()]
for a, b in zip(lines, lines[1:]):
    if a.startswith("Rep."):
        m = re.match(r"(.+?)\s*'s\s+(?:(\d+)\w*|At Large) District", b)
        name = re.sub(r",?\s*M\.D\.?", "", a[4:]).strip()
        name = re.sub(r"\s+", " ", name)
        rows.append(("Main Street", name, STATES[m.group(1).strip()], m.group(2) or "0", url))

# Squad (hardcoded)
for name, st in [("Alexandria Ocasio-Cortez","NY"),("Ilhan Omar","MN"),("Ayanna Pressley","MA"),("Rashida Tlaib","MI"),("Cori Bush","MO"),("Jamaal Bowman","NY"),("Summer Lee","PA"),("Delia Ramirez","IL"),("Greg Casar","TX")]:
    rows.append(("Squad", name, st, None, "hardcoded"))

raw = pd.DataFrame(rows, columns=["caucus","name","state","district","source_url"])
raw["name"] = raw["name"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip().str.replace("�", "'")

# ---------------- matching ----------------
def norm(s):
    s = re.sub(r"[-'’.]", " ", s).replace("�", "#")  # '#' = unknown char (Voteview mojibake)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z# ]", " ", s)
    return " ".join(w for w in s.split() if w not in {"jr","sr","ii","iii","iv"})

vv = pd.read_csv(os.path.join(REPO, "data/118/voteview/H118_members.csv"))
vv = vv[vv.chamber == "House"].drop_duplicates("bioguide_id")
vv["last"] = vv.bioname.str.split(",").str[0].map(norm)
rest = vv.bioname.str.split(",", n=1).str[1].fillna("")
vv["firsts"] = rest.map(lambda r: set(norm(re.sub(r"[()]", " ", r)).split()))

NICK = {"chuck":"charles","buddy":"earl","dusty":"dustin","jen":"jennifer","mike":"michael","andy":"andrew","greg":"gregory",
        "jim":"james","tom":"thomas","bill":"william","bob":"robert","dan":"daniel","dave":"david","don":"donald","ed":"edward",
        "nick":"nicholas","pete":"peter","chris":"christopher","joe":"joseph","ben":"benjamin","rick":"richard","rich":"richard"}

def match(r):
    c = vv[vv.state_abbrev == r.state]
    n = norm(r["name"]); words = n.split()
    def last_ok(L):
        lw = L.split()
        pat = [re.compile(x.replace("#", "[a-z]") + "$") for x in lw]
        return any(len(words[i:i+len(lw)]) == len(lw) and all(p.match(w) for p, w in zip(pat, words[i:i+len(lw)]))
                   for i in range(len(words)))
    cand = c[c["last"].map(last_ok)]
    if len(cand) == 0:  # e.g. "Jenniffer Gonzalez" vs "GONZALEZ-COLON": match first surname part
        cand = c[c["last"].map(lambda L: last_ok(L.split()[0]))]
    if len(cand) > 1 and pd.notna(r.district):
        cand2 = cand[cand.district_code.astype(str) == str(r.district)]
        if len(cand2): cand = cand2
    if len(cand) > 1:
        first = words[0]
        cand2 = cand[cand.firsts.map(lambda f: first in f or NICK.get(first) in f)]
        if len(cand2): cand = cand2
    if len(cand) > 1:  # prefer longest surname match (e.g. "Miller-Meeks" over "Miller")
        L = cand["last"].str.len(); cand = cand[L == L.max()]
    return cand.bioguide_id.iloc[0] if len(cand) == 1 else (None if len(cand) == 0 else "AMBIGUOUS")

raw["bioguide"] = raw.apply(match, axis=1)
raw[["caucus","name","state","district","source_url"]].to_csv(OUT_RAW, index=False)

bad = raw[raw.bioguide.isna() | (raw.bioguide == "AMBIGUOUS")]
print("UNMATCHED:\n", bad[["caucus","name","state","district","bioguide"]].to_string())
ok = raw[raw.bioguide.notna() & (raw.bioguide != "AMBIGUOUS")][["bioguide","caucus"]].drop_duplicates()
ok.to_csv(OUT, index=False)
print(ok.caucus.value_counts().to_string())
print("raw counts:\n", raw.caucus.value_counts().to_string())
# sanity: show matches for eyeballing
chk = raw.merge(vv[["bioguide_id","bioname","party_code"]], left_on="bioguide", right_on="bioguide_id", how="left")
chk[["caucus","name","state","bioname","party_code"]].to_csv(os.path.join(HERE, "match_check.csv"), index=False)
