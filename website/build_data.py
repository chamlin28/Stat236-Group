"""Build website/data.json for the project webpage (118th House, K=4, tau=1).

Inputs
  website/inputs/memberships_K4_tau1_118.csv   pi_1..pi_4 per member (from community_words branch / run_118.py)
  crec_parsed/parsed/crec_118.parquet          raw floor speeches (NOT committed; only word statistics leave this script)
  website/caucuses_118.csv                     bioguide, caucus
  website/committees_118.csv                   bioguide, committee (full House committees, 118th)
  data/118/...                                 votes (agreement matrix S, scree), voteview members (state, district)

Word statistics follow mixed_membership/community_words.py exactly (same documents, stopwords, vocabulary,
median-length rescaling, Fightin' Words with informative Dirichlet prior, one-member filter):
  * community words: pi-weighted counts, each community vs the rest, and within-party contrasts
  * member words:    each member's (rescaled) counts vs the rest of the House
  * word explorer:   rate per 10k words in each community for the most common terms

Run from the repo root:  uv run --with pyarrow python website/build_data.py
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "mixed_membership"))
import common as c  # noqa: E402

CONGRESS, K = 118, 4
CUTOFF = "2024-11-01"
MIN_MEMBERS, MIN_TOKENS, PRIOR, MAX_TOP_SHARE = 10, 1000, 0.01, 0.4   # as in community_words.py
N_COMMUNITY_WORDS, N_MEMBER_WORDS, N_EXPLORER = 80, 15, 3000
NGRAMS = {"words": (1, 1), "phrases": (1, 2)}

# --- stopwords: copied verbatim from community_words.py --------------------------------------------------
PROCEDURAL = """
speaker madam mr mrs ms gentleman gentlewoman gentlemen gentlewomen lady yield yields yielded yielding balance
time minutes minute seconds chair chairman chairwoman ranking member members colleague colleagues house floor
amendment amendments bill bills legislation act acts resolution rise today support urge vote votes voted
voting committee committees subcommittee consent unanimous clerk reclaiming rule rules motion pro tempore desk
chamber congress congressional district districts honor recognize recognized thank thanks sec section inquiry
parliamentary ordered question adopted rollcall objection remarks extend revise include extraneous material
ask asked going know want just like think said say says make let yes really got lot way come need did does
doesn don isn aren didn won wasn weren haven hasn couldn shouldn wouldn ll ve important year years day
pass passed passage mean things thing able sure work working good great tell look today point right
gentleman's speaker's h.r hr res bipartisan majority minority republican republicans democrat democrats
democratic party side aisle friend friends distinguished pre roll
""".split()
STATES = """
alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois
indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri
montana nebraska nevada new hampshire jersey mexico york north carolina dakota ohio oklahoma oregon
pennsylvania rhode island south tennessee texas utah vermont virginia washington west wisconsin wyoming
""".split()
STOP = sorted(set(ENGLISH_STOP_WORDS) | set(PROCEDURAL) | set(STATES))


def fightin_words(ya, yb, prior):
    na, nb, a0 = ya.sum(), yb.sum(), prior.sum()
    delta = np.log((ya + prior) / (na + a0 - ya - prior)) - np.log((yb + prior) / (nb + a0 - yb - prior))
    return delta / np.sqrt(1 / (ya + prior) + 1 / (yb + prior))


def top_share(C, w):
    contrib = sparse.csr_matrix(C.multiply(w[:, None]))
    total = np.asarray(contrib.sum(0)).ravel()
    return np.asarray(contrib.max(0).todense()).ravel() / np.maximum(total, 1e-12)


def r(x, d=3):
    return float(round(float(x), d))


# --- members ---------------------------------------------------------------------------------------------
m = pd.read_csv(HERE / "inputs/memberships_K4_tau1_118.csv")
vv = pd.read_csv(ROOT / f"data/{CONGRESS}/voteview/H{CONGRESS}_members.csv")
vv = vv[vv.chamber == "House"].drop_duplicates("bioguide_id").set_index("bioguide_id")
m["state"] = m.bioguide.map(vv.state_abbrev)
m["district"] = m.bioguide.map(vv.district_code)
P = m[[f"pi_{k+1}" for k in range(K)]].values
assert np.allclose(P.sum(1), 1), "pi rows must sum to 1"


def pretty(bioname):
    last, _, rest = bioname.partition(",")
    rest = rest.strip()
    nick = rest[rest.find("(") + 1:rest.find(")")] if "(" in rest else rest.split(" ")[0]
    suffix = " Jr." if rest.endswith("Jr.") else ""
    last = re.sub(r"\b(Mc|Mac(?=[a-z]{4}))([a-z])", lambda g: g.group(1) + g.group(2).upper(), last.title())
    return f"{nick} {last}{suffix}"


def congress_gov(name):
    """'Rep. Roy, Chip [R-TX-21]' -> 'Chip Roy' (Congress.gov names use the names members go by)."""
    core = re.sub(r"^Rep\.\s*|\s*\[.*\]$", "", name)
    last, _, first = core.partition(",")
    first = re.sub(r"\s+[A-Z]\.(?=\s|$)", "", first.strip())          # drop middle initials
    sfx = re.search(r",?\s*(Jr\.|Sr\.|II|III|IV)$", first)
    first = re.sub(r",?\s*(Jr\.|Sr\.|II|III|IV)$", "", first)
    return f"{first} {last.strip()}" + (f" {sfx.group(1)}" if sfx else "")


cg = pd.read_csv(ROOT / f"cosponsorship/data/{CONGRESS}/members.csv").drop_duplicates("bioguide").set_index("bioguide")
m["name"] = [congress_gov(cg.name[b]) if b in cg.index else pretty(bn) for b, bn in zip(m.bioguide, m.bioname)]
print(f"names from Congress.gov list: {m.bioguide.isin(cg.index).sum()} of {len(m)}")

# caucuses
cpath = HERE / "caucuses_118.csv"
cauc = pd.read_csv(cpath) if cpath.exists() else pd.DataFrame(columns=["bioguide", "caucus"])
epath = HERE / "caucuses_extra_118.csv"            # additional caucuses (RSC, CBC, CHC, ...)
if epath.exists():
    cauc = pd.concat([cauc, pd.read_csv(epath)]).drop_duplicates()
cmap = cauc.groupby("bioguide").caucus.apply(list).to_dict()
kpath = HERE / "committees_118.csv"
comm = pd.read_csv(kpath) if kpath.exists() else pd.DataFrame(columns=["bioguide", "committee"])
kmap = comm.groupby("bioguide").committee.apply(list).to_dict()
print(f"committee rows {len(comm)} ({comm.committee.nunique()} committees)")
dom_of = dict(zip(m.bioguide, P.argmax(1)))
# every filterable group: type -> name -> set of bioguide ids (members in the 428-member model)
groups = {"caucus": {}, "vertex": {}, "committee": {}}
for bg, cs in cmap.items():
    for cc in cs: groups["caucus"].setdefault(cc, set()).add(bg)
for bg, cs in kmap.items():
    for cc in cs: groups["committee"].setdefault(cc, set()).add(bg)
for bg, k in dom_of.items(): groups["vertex"].setdefault(f"V{k+1}", set()).add(bg)
inmodel = set(m.bioguide)
groups = {ty: {n: s & inmodel for n, s in g.items() if len(s & inmodel) >= 3} for ty, g in groups.items()}
group_words = {}
# member surnames are not "what they talk about" (colleagues are named in tributes and debate), so keep them out of
# the group and member word lists. The community lists stay exactly as in the report pipeline.
SURNAMES = set()
for nm, bn in zip(m.name, m.bioname):
    for part in re.split(r"[\s\-]+", nm.lower() + " " + bn.split(",")[0].lower()):
        part = re.sub(r"[^a-z]", "", part)
        if len(part) >= 3 and part not in ("jr", "iii"): SURNAMES.add(part)


def name_blocked(vocab, extra=()):
    ban = SURNAMES | set(extra)
    return np.array([any(tok in ban for tok in v.split()) for v in vocab])
missing = sorted(set(cauc.bioguide) - set(m.bioguide))
print(f"caucus rows {len(cauc)}; caucus members not in the 428-member sample: {len(missing)}")

# --- speeches -> member documents --------------------------------------------------------------------------
d = pd.read_parquet(ROOT / f"crec_parsed/parsed/crec_{CONGRESS}.parquet",
                    columns=["speech", "chamber", "is_extension", "date", "speaker_bioguide"])
d = d[(d.chamber == "H") & ~d.is_extension & (d.date < CUTOFF)
      & d.speaker_bioguide.isin(set(m.bioguide)) & (d.speech.str.len() >= 250)]
n_speeches = d.groupby("speaker_bioguide").size()
docs = d.groupby("speaker_bioguide").speech.apply(" \n ".join)
print(f"{len(d)} floor speeches from {len(docs)} of {len(m)} members")
m["n_speeches"] = m.bioguide.map(n_speeches).fillna(0).astype(int)

out_words = {}
member_words = {b: {} for b in m.bioguide}
explorer = None
comm_names = ["V1", "V2", "V3", "V4"]

for kind, ngram in NGRAMS.items():
    vec = CountVectorizer(token_pattern=r"(?u)\b[a-z]{3,}\b", stop_words=STOP, ngram_range=ngram,
                          min_df=MIN_MEMBERS, dtype=np.float64)
    C = vec.fit_transform(docs.values)
    vocab = vec.get_feature_names_out()
    L = np.asarray(C.sum(1)).ravel()
    n_single = np.asarray(C[:, [i for i, t in enumerate(vocab) if " " not in t]].sum(1)).ravel()
    keep = n_single >= MIN_TOKENS
    C, L, ids = C[keep], L[keep], docs.index[keep]
    C = sparse.csr_matrix(sparse.diags(np.median(L) / L) @ C)
    print(f"[{kind}] vocab {len(vocab)}, {keep.sum()} members with >= {MIN_TOKENS} words")
    mm = m.set_index("bioguide").loc[ids]
    pi = mm[[f"pi_{k+1}" for k in range(K)]].values
    prior = PRIOR * np.asarray(C.sum(0)).ravel()
    total = np.asarray(C.sum(0)).ravel()

    # community contrasts
    contrasts = [(f"V{k+1}", pi[:, k], 1 - pi[:, k]) for k in range(K)]
    isD, isR = (mm.party == "D").values.astype(float), (mm.party == "R").values.astype(float)
    contrasts += [("D_V1_vs_V2", pi[:, 0] * isD, pi[:, 1] * isD), ("R_V3_vs_V4", pi[:, 2] * isR, pi[:, 3] * isR)]
    tabs = {}
    for name, wa, wb in contrasts:
        ya, yb = np.asarray(C.T @ wa).ravel(), np.asarray(C.T @ wb).ravel()
        z = fightin_words(ya, yb, prior)
        ok = (top_share(C, wa) <= MAX_TOP_SHARE) & (top_share(C, wb) <= MAX_TOP_SHARE)
        ra, rb = 1e4 * ya / ya.sum(), 1e4 * yb / yb.sum()
        df = pd.DataFrame({"w": vocab, "z": z, "a": ra, "b": rb})[ok]
        pos = df[df.z > 1.96].nlargest(N_COMMUNITY_WORDS, "z")
        neg = df[df.z < -1.96].nsmallest(N_COMMUNITY_WORDS, "z")
        tabs[name] = {"pos": [[w, r(zz, 2), r(a, 2), r(b, 2)] for w, zz, a, b in pos.values],
                      "neg": [[w, r(-zz, 2), r(b, 2), r(a, 2)] for w, zz, a, b in neg.values]}
        print(f"  {name:12s} + {', '.join(pos.w.head(8))}")
    out_words[kind] = tabs

    blocked_vocab = name_blocked(vocab)
    # member words: member vs rest of the House
    for i, b in enumerate(ids):
        ya = C[i].toarray().ravel()
        yb = total - ya
        z = fightin_words(ya, yb, prior)
        z[ya < 3] = -np.inf                                      # must actually use the word a few times
        z[blocked_vocab] = -np.inf                               # not colleagues' surnames
        top = np.argsort(-z)[:N_MEMBER_WORDS]
        top = [j for j in top if z[j] > 1.96]
        member_words[b][kind] = [[vocab[j], r(z[j], 1), r(1e4 * ya[j] / ya.sum(), 2), r(1e4 * yb[j] / yb.sum(), 2)]
                                 for j in top]

    # filterable groups: members of the group vs everyone else (single words only)
    if kind == "words":
        idx_of = {bg: i for i, bg in enumerate(ids)}
        for ty, g in groups.items():
            for name, mem_ids in g.items():
                inn = np.zeros(len(ids))
                for bg in mem_ids:
                    if bg in idx_of: inn[idx_of[bg]] = 1.0
                if inn.sum() < 3: continue
                ya, yb = np.asarray(C.T @ inn).ravel(), np.asarray(C.T @ (1 - inn)).ravel()
                z = fightin_words(ya, yb, prior)
                ok = (top_share(C, inn) <= MAX_TOP_SHARE) & (top_share(C, 1 - inn) <= MAX_TOP_SHARE)
                own = re.findall(r"[a-z]{4,}", name.lower()) if ty == "committee" else []
                ok &= ~name_blocked(vocab, own)
                ra, rb = 1e4 * ya / ya.sum(), 1e4 * yb / yb.sum()
                df = pd.DataFrame({"w": vocab, "z": z, "a": ra, "b": rb})[ok]
                pos = df[df.z > 1.96].nlargest(60, "z")
                group_words[f"{ty}|{name}"] = {"n": int(len(mem_ids)), "nw": int(inn.sum()),
                    "pos": [[w, r(zz, 2), r(a_, 2), r(b_, 2)] for w, zz, a_, b_ in pos.values]}

    # word explorer (single words only): rate per 10k words in each community and overall
    if kind == "words":
        Y = np.asarray((C.T @ pi)).T                              # K x V
        rates = 1e4 * Y / Y.sum(1, keepdims=True)
        overall = 1e4 * total / total.sum()
        idx = np.argsort(-total)[:N_EXPLORER]
        explorer = {vocab[j]: [r(x, 2) for x in rates[:, j]] + [r(overall[j], 2)] for j in idx}

# --- agreement matrix & scree ------------------------------------------------------------------------------
n_bills = len(pd.read_csv(ROOT / f"data/{CONGRESS}/bills.csv"))
V, mem = c.load_votes(CONGRESS, min_votes=int(np.ceil(n_bills / 2)))
S = c.agreement_matrix(V)
lam = np.linalg.eigvalsh(S)
lam = lam[np.argsort(-np.abs(lam))][:20]
# heatmap ordered by party then dominant community then pi; 2x2 block averages to keep the file small
pos_of = {b: i for i, b in enumerate(mem.bioguide_id.values)}
mi = m.set_index("bioguide")
order_b = sorted(mi.index, key=lambda b: (int(np.argmax(P[m.index[m.bioguide == b][0]])),
                                           -P[m.index[m.bioguide == b][0]].max()))
ordr = [pos_of[b] for b in order_b if b in pos_of]
Ss = S[np.ix_(ordr, ordr)].copy()
np.fill_diagonal(Ss, 1)
n2 = len(ordr) // 2 * 2
Sb = Ss[:n2, :n2].reshape(n2 // 2, 2, n2 // 2, 2).mean((1, 3))
dom_sorted = [int(np.argmax(mi.loc[b, [f"pi_{k+1}" for k in range(K)]].values)) for b in order_b if b in pos_of]
bounds = [i for i in range(1, len(dom_sorted)) if dom_sorted[i] != dom_sorted[i - 1]]

# --- vertices ----------------------------------------------------------------------------------------------
vert = []
for k in range(K):
    w = P[:, k] + 1e-12
    near = np.argsort(-P[:, k])[:8]
    vert.append({"id": f"V{k+1}", "d_share": r(np.average(m.party == "D", weights=w), 3),
                 "nom": r(np.average(m.nominate_dim1.fillna(0), weights=w), 3),
                 "mass": r(P[:, k].sum(), 1), "nearest": [m.name[i] for i in near]})

members = []
for i, row in m.iterrows():
    members.append({"b": row.bioguide, "n": row["name"], "p": row.party, "s": row.state,
                    "d": int(row.district) if pd.notna(row.district) else None,
                    "nom": r(row.nominate_dim1, 3) if pd.notna(row.nominate_dim1) else None,
                    "pi": [r(x, 4) for x in P[i]], "sp": int(row.n_speeches),
                    "c": cmap.get(row.bioguide, []), "k": kmap.get(row.bioguide, []), "w": member_words[row.bioguide]})

data = {"meta": {"n_members": len(m), "n_bills": n_bills, "n_speeches": int(len(d)),
                 "n_with_words": sum(1 for x in members if x["w"]), "pure_share": r((P.max(1) > .9).mean(), 3)},
        "vertices": vert, "members": members, "words": out_words, "explorer": explorer, "groups": {ty: {n: sorted(s) for n, s in g.items()} for ty, g in groups.items()}, "group_words": group_words,
        "scree": [r(x, 2) for x in lam],
        "heat": {"z": (np.round(Sb * 100)).astype(int).tolist(), "bounds": [b // 2 for b in bounds]}}
(HERE / "data.json").write_text(json.dumps(data, separators=(",", ":")))
print(f"wrote data.json ({(HERE / 'data.json').stat().st_size / 1e6:.2f} MB); members with words "
      f"{data['meta']['n_with_words']}; pure share {data['meta']['pure_share']}; scree {data['scree'][:5]}")

# --- inline the data into the page -----------------------------------------------------------------------------
tpl = (HERE / "template.html").read_text(encoding="utf-8")
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
# standalone document, so it works when uploaded to the course site as-is
page = head + tpl.replace("__DATA__", blob) + "\n</html>\n"
# Plotly and wordcloud2 are inlined (from website/vendor), so the page works with no network and in locked-down browsers
for token, fname in (("__PLOTLY__", "plotly.min.js"), ("__WORDCLOUD__", "wordcloud2.min.js")):
    js = (HERE / "vendor" / fname).read_text(encoding="utf-8").replace("</script", "<\\/script")
    i = page.index(token)
    page = page[:i] + js + page[i + len(token):]
(HERE / "index.html").write_text(page, encoding="utf-8")
print(f"wrote index.html ({(HERE / 'index.html').stat().st_size / 1e6:.2f} MB)")
