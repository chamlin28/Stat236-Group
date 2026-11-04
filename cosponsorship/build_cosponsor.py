"""Cosponsorship network of House members, kept separate from the vote graph in data/<congress>/ (the whole cosponsorship/ folder is a side project).

Two lawmakers are linked when they both took part in the same bill: as its sponsor or as a (non-withdrawn)
cosponsor. Uses ALL House bills and resolutions (H.R., H.Res., H.J.Res., H.Con.Res.), not only the ones that got a
final-passage vote, from the GovInfo bulk BILLSTATUS files (no API key needed).

Writes to cosponsorship/data/<congress>/:
  members.csv        bioguide, name, party, state, district, n_sponsored, n_cosponsored
  bills.csv          one row per bill: sponsor, cosponsor counts by party, policy area, became_law, ...
  sponsorship.csv    bipartite edge list: bill, member, role (sponsor/cosponsor), original, date, withdrawn
  edges.csv          member--member projection (a < b): weight = # shared bills, weight_norm = sum of 1/(k-1) over
                     shared bills (k = participants; discounts bills with huge cosponsor lists), weight_idf =
                     sum of log(N_members / k) over shared bills (smoother discount, ~0 for bills nearly everyone
                     signs), plus a_sponsored_b_cosponsored / b_sponsored_a_cosponsored (directed support counts)
  graph.graphml      the projection as a networkx graph (nodes carry party/state/district); edge `weight` = weight_idf,
                     `count` = raw shared bills, `weight_norm` = the 1/(k-1) version

    uv run python -I cosponsorship/build_cosponsor.py --congress 115 116 117 118
Raw zips go to cosponsorship/data/raw/<congress>/ (one folder each) and are read in place, never extracted.
"""
import argparse
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "data"
TYPES = ["hr", "hres", "hjres", "hconres"]
URL = "https://www.govinfo.gov/bulkdata/BILLSTATUS/{c}/{t}/BILLSTATUS-{c}-{t}.zip"


def download(c):
    folder = ROOT / "raw" / str(c)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in TYPES:
        f = folder / f"BILLSTATUS-{c}-{t}.zip"
        if not f.exists():
            print(f"  downloading {f.name} ...")
            urllib.request.urlretrieve(URL.format(c=c, t=t), f)
        paths.append(f)
    return paths


def person(item):
    return {"bioguide": item.findtext("bioguideId"), "name": item.findtext("fullName"), "party": item.findtext("party"),
            "state": item.findtext("state"), "district": item.findtext("district")}


def parse(zip_paths):
    bills, spons, people = [], [], {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if not n.endswith(".xml"):
                    continue
                b = ET.fromstring(z.read(n)).find("bill")
                if b is None:
                    continue
                # a few records use the alternate tags billType / billNumber (e.g. H.R. 3354 in the 115th)
                btype, bnum = b.findtext("type") or b.findtext("billType"), b.findtext("number") or b.findtext("billNumber")
                if not (btype and bnum):
                    print(f"  skipping {n}: no bill type/number")
                    continue
                bill = f"{btype}{bnum}"
                s = b.find("sponsors/item")
                sp = person(s) if s is not None else None
                rows = []
                if sp and sp["bioguide"]:
                    people[sp["bioguide"]] = sp
                    rows.append({"bill": bill, "member": sp["bioguide"], "role": "sponsor", "original": True,
                                 "date": b.findtext("introducedDate"), "withdrawn": False})
                for it in b.findall("cosponsors/item"):
                    p = person(it)
                    if not p["bioguide"]:
                        continue
                    people.setdefault(p["bioguide"], p)
                    rows.append({"bill": bill, "member": p["bioguide"], "role": "cosponsor",
                                 "original": it.findtext("isOriginalCosponsor") == "True",
                                 "date": it.findtext("sponsorshipDate"), "withdrawn": bool(it.findtext("sponsorshipWithdrawnDate"))})
                spons += rows
                bills.append({"bill": bill, "type": btype, "number": int(bnum),
                              "introduced": b.findtext("introducedDate"), "title": b.findtext("title"),
                              "policy_area": b.findtext("policyArea/name"), "sponsor": sp["bioguide"] if sp else None,
                              "became_law": b.find("laws/item") is not None})
    return pd.DataFrame(bills), pd.DataFrame(spons), pd.DataFrame(people.values()).set_index("bioguide")


def main(c):
    print(f"[{c}] bulk files")
    bills, spons, people = parse(download(c))
    live = spons[~spons.withdrawn]                               # withdrawn cosponsors do not count
    members = sorted(live.member.unique())
    mi = {m: i for i, m in enumerate(members)}
    bi = {b: j for j, b in enumerate(bills.bill)}

    B = np.zeros((len(members), len(bills)), dtype=np.float32)   # participation (sponsor or cosponsor)
    S = np.zeros_like(B); C = np.zeros_like(B)                   # sponsor / cosponsor separately
    for r in live.itertuples():
        i, j = mi[r.member], bi[r.bill]
        B[i, j] = 1; (S if r.role == "sponsor" else C)[i, j] = 1
    k = B.sum(0)
    W = B @ B.T
    Wn = (B / np.maximum(k - 1, 1)) @ B.T
    # IDF weight: a bill that nearly every member signed (k close to N) tells us little about who is close to whom.
    idf = np.log(len(members) / np.maximum(k, 1))
    Wi = (B * idf) @ B.T
    SC = S @ C.T                                                 # SC[i, j] = # bills i sponsored that j cosponsored
    np.fill_diagonal(W, 0); np.fill_diagonal(Wn, 0); np.fill_diagonal(Wi, 0)

    iu, ju = np.triu_indices(len(members), 1)
    keep = W[iu, ju] > 0
    a, b_ = iu[keep], ju[keep]
    edges = pd.DataFrame({"a": np.array(members)[a], "b": np.array(members)[b_], "weight": W[a, b_].astype(int),
                          "weight_norm": Wn[a, b_].round(5), "weight_idf": Wi[a, b_].round(4),
                          "a_sponsored_b_cosponsored": SC[a, b_].astype(int),
                          "b_sponsored_a_cosponsored": SC[b_, a].astype(int)})

    out = ROOT / str(c); out.mkdir(parents=True, exist_ok=True)
    bills = bills.assign(n_participants=k.astype(int))
    pc = live[live.role == "cosponsor"].merge(people[["party"]], left_on="member", right_index=True)
    for p in ["D", "R"]:
        bills[f"n_cosponsors_{p}"] = bills.bill.map(pc[pc.party == p].groupby("bill").size()).fillna(0).astype(int)
    bills["n_cosponsors"] = bills.bill.map(pc.groupby("bill").size()).fillna(0).astype(int)
    bills["bipartisan"] = (bills.n_cosponsors_D > 0) & (bills.n_cosponsors_R > 0)
    mem = people.loc[members].copy()
    mem["n_sponsored"] = pd.Series(S.sum(1), index=members).astype(int)
    mem["n_cosponsored"] = pd.Series(C.sum(1), index=members).astype(int)
    mem.rename_axis("bioguide").reset_index().to_csv(out / "members.csv", index=False)
    bills.to_csv(out / "bills.csv", index=False)
    spons.to_csv(out / "sponsorship.csv", index=False)
    edges.to_csv(out / "edges.csv", index=False)

    G = nx.Graph()
    G.add_nodes_from((m, {k_: ("" if pd.isna(v) else v) for k_, v in mem.loc[m, ["name", "party", "state", "district"]].items()}) for m in members)
    # In the graph file, `weight` (what networkx algorithms use by default) is the IDF-discounted weight;
    # the raw shared-bill count is kept as `count`.
    G.add_edges_from((r.a, r.b, {"weight": float(r.weight_idf), "count": int(r.weight), "weight_norm": float(r.weight_norm)})
                     for r in edges.itertuples())
    nx.write_graphml(G, out / "graph.graphml")

    same = edges.merge(mem.party.rename("pa"), left_on="a", right_index=True).merge(mem.party.rename("pb"), left_on="b", right_index=True)
    cross = same[same.pa != same.pb]
    print(f"[{c}] {len(bills)} bills ({bills.bipartisan.sum()} bipartisan, {bills.became_law.sum()} became law), "
          f"{len(members)} members, {len(edges)} member pairs; "
          f"{len(cross) / len(same):.1%} of pairs and {cross.weight.sum() / same.weight.sum():.1%} of weight are cross-party")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--congress", type=int, nargs="+", default=[118])
    for cong in ap.parse_args().congress:
        main(cong)
