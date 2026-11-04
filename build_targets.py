"""Build the election-outcome target table: one row per House member per Congress.

Congress N's members face the general election in year 2018 + 2 * (N - 115).
Input : data/elections/1976-2024-house.tab (MIT Election Lab) and data/<N>/voteview/H<N>_members.csv
Output: data/elections/targets.csv

    uv run python build_targets.py
"""
import io
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

CONGRESSES = [115, 116, 117, 118]      # 119th's election has not happened yet
ELECTIONS = Path("data/elections/1976-2024-house.tab")
PARTY = {"DEMOCRAT": "D", "DEMOCRATIC-FARMER-LABOR": "D", "REPUBLICAN": "R"}


def election_year(congress):
    return 2018 + 2 * (congress - 115)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z ]", " ", s.replace("-", " ")).split()


def load_results():
    # The file mixes quote-escaping styles (\"" inside quoted names); flatten them before parsing.
    raw = ELECTIONS.read_text(encoding="utf-8-sig").replace('\\""', "'")
    e = pd.read_csv(io.StringIO(raw), low_memory=False)
    for c in ["special", "runoff", "writein"]:
        e[c] = e[c].astype(str).str.upper().eq("TRUE")
    e = e[(e.stage == "GEN") & ~e.special & ~e.runoff & ~e.writein].copy()
    e["candidate"] = e["candidate"].astype(str)
    # Non-candidate rows (Maine ranked-choice "exhausted ballots", blank/scattering/over/under votes).
    e = e[~e.candidate.str.upper().str.contains(r"EXHAUSTED|BLANK|SCATTER|OVERVOTE|UNDERVOTE|VOID|^NA$", regex=True)]
    # Fusion tickets list one candidate several times (one row per party line): sum their votes.
    e["pty"] = e.party.map(PARTY).fillna("O")
    g = (e.groupby(["year", "state_po", "district", "candidate"], as_index=False)
          .agg(votes=("candidatevotes", "sum"), total=("totalvotes", "max"),
               pty=("pty", lambda s: "D" if "D" in set(s) else "R" if "R" in set(s) else "O")))
    return g


def match(row, pool):
    """Best candidate in pool (one state-year) for a Voteview member, or None."""
    last, _, first = row.bioname.partition(",")
    last = norm(last)
    first = norm(re.sub(r"\(.*?\)", " ", first))
    nick = [t for n in re.findall(r"\((.*?)\)", row.bioname) for t in norm(n)]
    last_sq = "".join(last)
    best, best_score = None, 0
    for r in pool.itertuples():
        ctoks = norm(r.candidate)
        toks = set(ctoks)
        # Surname = a run of candidate tokens equal to it with punctuation removed (D'Esposito, Diaz-Balart);
        # the file has mangled non-ASCII characters, which norm() drops on both sides.
        if not any("".join(ctoks[i:j]) == last_sq for i in range(len(ctoks)) for j in range(i + 1, len(ctoks) + 1)):
            continue
        same_seat = r.district == row.district_code and r.pty == row.pty
        named = (any(t in toks for t in first) or any(n in toks for n in nick)
                 or (first and any(first[0][:3] == t[:3] for t in toks)))
        # A shared surname alone is not enough (Jim vs Daniel Cooper), unless it is also the same
        # seat and party (covers nicknames the roster lacks: Chip Roy = Charles, Tom Emmer = Thomas).
        if not (named or same_seat):
            continue
        score = 1 + 2 * bool(named) + (r.district == row.district_code) + (r.pty == row.pty)
        if score > best_score:
            best, best_score = r, score
    return best


def outcome(row, g):
    """General-election outcome of this member in year `row.year`, from results table g."""
    pool_all = g[(g.year == row.year) & (g.state_po == row.state_abbrev)]
    hit = match(row, pool_all)
    if hit is None:
        return {}
    dist = pool_all[pool_all.district == hit.district]
    others = dist[dist.candidate != hit.candidate].votes.fillna(0)
    if len(others) == 0:      # uncontested: some states (e.g. FL) report no vote counts at all
        return {"cand": hit.candidate, "res_district": hit.district, "vote_share": 1.0, "margin": 1.0,
                "won": True, "uncontested": True, "two_party_share": np.nan}
    total = dist.votes.sum()
    top_opp = others.max()
    # Two-party share = own party / (D + R). Undefined without a D and an R on the ballot
    # (uncontested, minor-party opponent only, or a same-party top-two race).
    dv, rv = (dist[dist.pty == p].votes.sum() for p in ("D", "R"))
    own, tp = (dv, dv + rv) if row.pty == "D" else (rv, dv + rv)
    two_party = own / tp if row.pty in ("D", "R") and dv > 0 and rv > 0 else np.nan
    return {"cand": hit.candidate, "res_district": hit.district, "vote_share": hit.votes / total,
            "margin": (hit.votes - top_opp) / total, "won": bool(hit.votes > top_opp), "uncontested": False,
            "two_party_share": two_party}


def national_dem_share(g):
    """Actual national D/(D+R) House vote per year, over districts where both parties ran (uncontested seats
    often report no votes, so including them would bias the total)."""
    d = g[g.pty.isin(["D", "R"])].pivot_table(index=["year", "state_po", "district"], columns="pty",
                                              values="votes", aggfunc="sum")
    d = d[(d.D > 0) & (d.R > 0)]
    s = d.groupby("year").sum()
    return s.D / (s.D + s.R)


def main():
    g = load_results()
    rows = []
    for c in CONGRESSES:
        m = pd.read_csv(f"data/{c}/voteview/H{c}_members.csv")
        m = m[(m.chamber == "House") & m.bioguide_id.notna()].copy()
        m["pty"] = m.party_code.map({100: "D", 200: "R"}).fillna("O")
        y = election_year(c)
        for r in m.itertuples():
            out = outcome(pd.Series({**r._asdict(), "year": y}), g)
            prev = outcome(pd.Series({**r._asdict(), "year": y - 2}), g)
            rows.append({"congress": c, "election_year": y, "bioguide": r.bioguide_id, "name": r.bioname,
                         "state": r.state_abbrev, "district": r.district_code, "party": r.pty,
                         "ran": bool(out), **out,
                         "prev_vote_share": prev.get("vote_share", np.nan), "prev_margin": prev.get("margin", np.nan),
                         "prev_two_party_share": prev.get("two_party_share", np.nan),
                         "prev_res_district": prev.get("res_district", np.nan)})
    t = pd.DataFrame(rows)

    # National environment from the member's side: own party's national two-party share, this year and last.
    nat = national_dem_share(g)
    own = lambda yr: np.where(t.party == "D", t.election_year.map(lambda y: nat[yr(y)]),
                              t.election_year.map(lambda y: 1 - nat[yr(y)]))
    t["national_share"], t["prev_national_share"] = own(lambda y: y), own(lambda y: y - 2)
    # tilt = change in own two-party share minus change in the party's national share (uniform-swing residual).
    t["tilt"] = (t.two_party_share - t.prev_two_party_share) - (t.national_share - t.prev_national_share)
    print("national D two-party share:", nat.loc[2016:].round(4).to_dict())
    # Members appear once per Congress in Voteview, but some seats turned over mid-term (two members, one seat).
    for col in ["won", "uncontested"]:
        t[col] = t[col].astype("boolean")
    t.to_csv("data/elections/targets.csv", index=False)

    print(t.groupby("congress").agg(members=("bioguide", "size"), ran=("ran", "sum"), won=("won", "sum"),
                                    uncontested=("uncontested", "sum"), mean_share=("vote_share", "mean")).round(3))
    print("\nlost general election, by party:")
    print(t[t.won == False].groupby(["congress", "party"]).size().unstack(fill_value=0))
    print("\nlowest-confidence matches (winner name differs from member's last name is impossible by construction;",
          "check shares outside [0.3, 1]):")
    print(t[(t.vote_share < 0.3)][["congress", "name", "cand", "vote_share"]].to_string())
    print("\nnot found in general-election results (retired / lost primary / other office / name mismatch):")
    print(t[~t.ran].groupby("congress").size().to_dict())


if __name__ == "__main__":
    main()
