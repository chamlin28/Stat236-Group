"""Words that distinguish the mixed-membership communities (118th House), word clouds, and an eigen-map.

Method
  1. One document per member: their House floor speeches in the 118th Congress (speaker_bioguide set,
     >= 250 characters, not Extensions of Remarks, dated before 2024-11-01, as in the speech features).
  2. Member x term counts (unigrams + bigrams, >= MIN_MEMBERS members use the term). Stopwords include floor
     procedure ("yield", "gentleman", "reclaiming my time", ...) and state names. Each member's counts are
     rescaled to the median length, so every member counts once and heavy speakers do not dominate.
  3. Community counts are pi-weighted sums (soft membership): y_k = sum_i pi_ik * counts_i.
  4. Distinctive words: weighted log-odds with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008,
     "Fightin' Words"). z > 0: used more by group A; z < 0: used more by group B.
     Words where one member supplies > MAX_TOP_SHARE of a group's usage are dropped (one-member words).
  Contrasts: each community vs the rest, and within each party the two largest communities (e.g. Freedom
  Caucus vs mainstream Republicans), which removes the plain party difference.

Uses the tau=1 memberships (votes + speech PCs). Caveat: those communities were built partly from speech,
so part of the word difference is by construction. Run twice: words + phrases, and single words only (_words).

Run from the repo root (after run_118.py):  uv run python mixed_membership/community_words.py
Outputs go to mixed_membership/words/."""
import sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import ConnectionPatch
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from wordcloud import WordCloud
sys.path.insert(0, str(Path(__file__).parent))
import common as c

CONGRESS = 118
TAUS = (1.0,)              # tau=1: votes + speech PCs (the memberships in the README)
NGRAMS = {"": (1, 2), "_words": (1, 1)}   # file suffix -> n-gram range: words + phrases, words only
KS = (3, 4)
CUTOFF = "2024-11-01"
MIN_MEMBERS = 10           # a term must be used by at least this many members
MIN_TOKENS = 1000          # members with fewer kept tokens are dropped (too little text to rescale)
PRIOR = 0.01               # Dirichlet prior: alpha_w = PRIOR * (total count of w)
MAX_TOP_SHARE = 0.4        # drop a word if one member supplies more than this share of a group's usage
N_CLOUD = 40               # words per cloud
HERE = Path(__file__).parent
OUT = HERE / "words"
OUT.mkdir(exist_ok=True)

COLORS = {3: ["#2166ac", "#7b3294", "#b2182b"], 4: ["#2166ac", "#4393c3", "#d6604d", "#b2182b"]}

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


def member_documents(bioguides):
    d = pd.read_parquet(c.ROOT / f"crec_parsed/parsed/crec_{CONGRESS}.parquet",
                        columns=["speech", "chamber", "is_extension", "date", "speaker_bioguide"])
    d = d[(d.chamber == "H") & ~d.is_extension & (d.date < CUTOFF)
          & d.speaker_bioguide.isin(bioguides) & (d.speech.str.len() >= 250)]
    print(f"{len(d)} floor speeches from {d.speaker_bioguide.nunique()} of {len(bioguides)} members")
    return d.groupby("speaker_bioguide").speech.apply(" \n ".join)


def fightin_words(ya, yb, prior):
    """Monroe, Colaresi & Quinn (2008) log-odds with informative Dirichlet prior. Returns z (A minus B)."""
    a = prior
    na, nb, a0 = ya.sum(), yb.sum(), a.sum()
    delta = (np.log((ya + a) / (na + a0 - ya - a)) - np.log((yb + a) / (nb + a0 - yb - a)))
    return delta / np.sqrt(1 / (ya + a) + 1 / (yb + a))


def top_share(C, w):
    """Largest single-member share of the w-weighted usage of each term."""
    contrib = sparse.csr_matrix(C.multiply(w[:, None]))
    total = np.asarray(contrib.sum(0)).ravel()
    return np.asarray(contrib.max(0).todense()).ravel() / np.maximum(total, 1e-12)


def contrasts(m, K):
    """(name, weight_A, weight_B, label_A, label_B, k_A, k_B). Every community vs the rest, then within party."""
    pi = m[[f"pi_{k+1}" for k in range(K)]].values
    out = [(f"V{k+1}_vs_rest", pi[:, k], 1 - pi[:, k], f"V{k+1}", "rest", k, None) for k in range(K)]
    for p in ("D", "R"):
        inp = (m.party == p).values.astype(float)
        mass = (pi * inp[:, None]).sum(0)
        a, b = np.argsort(-mass)[:2]
        if mass[b] < 10:                                    # fewer than ~10 effective members: skip
            continue
        a, b = sorted((a, b))                               # left (lower NOMINATE) community first
        out.append((f"{p}_V{a+1}_vs_V{b+1}", pi[:, a] * inp, pi[:, b] * inp,
                    f"{p} in V{a+1}", f"{p} in V{b+1}", a, b))
    return out


def cloud(words, color, width=900, height=520):
    wc = WordCloud(width=width, height=height, background_color="white", prefer_horizontal=1.0,
                   max_words=N_CLOUD, random_state=0, color_func=lambda *a, **k: color)
    return wc.generate_from_frequencies(words).to_array()


def top_words(df, sign, n=N_CLOUD):
    d = df[df.z * sign > 1.96].sort_values("z", ascending=sign < 0).head(n)
    return dict(zip(d.word, d.z.abs()))


def vertex_label(m, K, k):
    pi = m[f"pi_{k+1}"].values
    d = np.average(m.party == "D", weights=pi + 1e-12)
    nom = np.average(m.nominate_dim1.fillna(0), weights=pi + 1e-12)
    near = ", ".join(m.bioname.values[i].split(",")[0].title() for i in np.argsort(-pi)[:3])
    return f"V{k+1}: D {d*100:.0f}%, NOM1 {nom:+.2f}\n({near})"


# ---------------------------------------------------------------- counts (shared by every K and tau)
m0 = pd.read_csv(HERE / f"memberships_K3_tau1_{CONGRESS}.csv")
docs = member_documents(set(m0.bioguide))
for suffix, ngram in NGRAMS.items():
    vec = CountVectorizer(token_pattern=r"(?u)\b[a-z]{3,}\b", stop_words=STOP, ngram_range=ngram,
                          min_df=MIN_MEMBERS, dtype=np.float64)
    C_all = vec.fit_transform(docs.values)
    vocab = vec.get_feature_names_out()
    L = np.asarray(C_all.sum(1)).ravel()
    n_words = np.asarray(C_all[:, [i for i, t in enumerate(vocab) if " " not in t]].sum(1)).ravel()
    keep = n_words >= MIN_TOKENS                            # single-word count, so both runs keep the same members
    C_all, L, kept_ids = C_all[keep], L[keep], docs.index[keep]
    C_all = sparse.diags(np.median(L) / L) @ C_all              # every member counts once
    print(f"[{suffix or 'phrases'}] vocabulary {len(vocab)} terms; {keep.sum()} members with >= {MIN_TOKENS} tokens "
          f"(dropped {(~keep).sum()}); median length {np.median(L):.0f} tokens")

    # ---------------------------------------------------------------- per K and tau
    for tau in TAUS:
        for K in KS:
            m = pd.read_csv(HERE / f"memberships_K{K}_tau{tau:g}_{CONGRESS}.csv").set_index("bioguide")
            m = m.loc[m.index.intersection(kept_ids)]
            C = C_all[[kept_ids.get_loc(b) for b in m.index]]
            m = m.reset_index()
            prior = PRIOR * np.asarray(C.sum(0)).ravel()
            tag = f"K{K}_tau{tau:g}{suffix}"
            kind = "single words" if ngram == (1, 1) else "words + phrases"
            rows, results = [], {}
            for name, wa, wb, la, lb, ka, kb in contrasts(m, K):
                ya = np.asarray(C.T @ wa).ravel(); yb = np.asarray(C.T @ wb).ravel()
                z = fightin_words(ya, yb, prior)
                ok = (top_share(C, wa) <= MAX_TOP_SHARE) & (top_share(C, wb) <= MAX_TOP_SHARE)
                df = pd.DataFrame({"contrast": name, "word": vocab, "z": z,
                                   "per10k_A": 1e4 * ya / ya.sum(), "per10k_B": 1e4 * yb / yb.sum()})[ok]
                results[name] = (df, la, lb, ka, kb)
                rows.append(df[df.z.abs() > 1.96])
                print(f"[{tag}] {name}:  A+ {', '.join(df.nlargest(8, 'z').word)}")
                print(f"[{tag}] {' ' * len(name)}   B+ {', '.join(df.nsmallest(8, 'z').word)}")
            pd.concat(rows).sort_values(["contrast", "z"], ascending=[True, False]).to_csv(
                OUT / f"words_{tag}.csv", index=False)

            # each community vs the rest: one cloud per community
            fig, axes = plt.subplots(1, K, figsize=(5.2 * K, 4.2))
            for k, ax in enumerate(axes):
                df = results[f"V{k+1}_vs_rest"][0]
                ax.imshow(cloud(top_words(df, +1), COLORS[K][k])); ax.axis("off")
                ax.set_title(vertex_label(m, K, k), fontsize=10, color=COLORS[K][k])
            fig.suptitle(f"{CONGRESS}th House, K={K}, tau={tau:g}, {kind}: words used more by each community than by the rest "
                         f"(size = Fightin' Words z)", fontsize=11)
            plt.tight_layout(); plt.savefig(OUT / f"clouds_{tag}.png", dpi=130); plt.close()

            # within-party contrasts: A-side and B-side clouds
            within = [r for r in results if not r.endswith("_vs_rest")]
            if within:
                fig, axes = plt.subplots(len(within), 2, figsize=(11, 4.2 * len(within)), squeeze=False)
                for row, name in zip(axes, within):
                    df, la, lb, ka, kb = results[name]
                    for ax, sign, lab, k in ((row[0], +1, la, ka), (row[1], -1, lb, kb)):
                        ax.imshow(cloud(top_words(df, sign), COLORS[K][k])); ax.axis("off")
                        other = lb if sign > 0 else la
                        ax.set_title(f"{lab}: more than {other}", fontsize=10, color=COLORS[K][k])
                fig.suptitle(f"{CONGRESS}th House, K={K}, tau={tau:g}, {kind}: within-party contrasts", fontsize=11)
                plt.tight_layout(); plt.savefig(OUT / f"clouds_within_party_{tag}.png", dpi=130); plt.close()

            # eigen-map (K=3): members at SCORE coordinates of S, coloured by their pi mix, clouds at the vertices
            if K == 3:
                xy = m[["score_1", "score_2"]].values
                pi = m[["pi_1", "pi_2", "pi_3"]].values
                rgb = pi @ np.array([matplotlib.colors.to_rgb(h) for h in COLORS[3]])
                fig, ax = plt.subplots(figsize=(13, 10))
                ax.scatter(*xy.T, c=rgb, s=28, alpha=.85, edgecolor="white", lw=.4)
                centre = xy.mean(0)
                span = xy.max(0) - xy.min(0)
                boxes = []
                for k in range(3):
                    vx = xy[np.argsort(-pi[:, k])[:10]].mean(0)            # where the purest members sit
                    d = (vx - centre) / (np.linalg.norm((vx - centre) / span) + 1e-12)
                    box = vx + .8 * d
                    boxes.append(box)
                    img = cloud(top_words(results[f"V{k+1}_vs_rest"][0], +1), COLORS[3][k], 700, 420)
                    ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=.42), vx, xybox=box, boxcoords="data",
                                                 frameon=True, bboxprops=dict(ec=COLORS[3][k], lw=1.5),
                                                 arrowprops=dict(arrowstyle="-", color=COLORS[3][k], lw=1.2)))
                    ax.annotate(vertex_label(m, 3, k).split("\n")[0], box, xytext=(0, 105),
                                textcoords="offset points", ha="center", fontsize=10, color=COLORS[3][k],
                                fontweight="bold")
                allp = np.vstack([xy, boxes])
                lo, hi = allp.min(0), allp.max(0)
                pad = .25 * (hi - lo)
                ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
                ax.set_xlabel("SCORE coordinate 1 (xi_2 / xi_1)"); ax.set_ylabel("SCORE coordinate 2 (xi_3 / xi_1)")
                ax.set_title(f"{CONGRESS}th House eigen-map: members (colour = membership mix, tau={tau:g}) "
                             f"and each community's distinctive words ({kind})", fontsize=11)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                plt.tight_layout(); plt.savefig(OUT / f"eigenmap_{tag}.png", dpi=130); plt.close()

                # simplex (K=3): same triangle as run_118.py, members coloured by party, clouds at the corners
                vt = np.array([[0, 0], [1, 0], [.5, np.sqrt(3) / 2]])
                P = pi @ vt + np.random.default_rng(0).normal(scale=.003, size=(len(pi), 2))
                pcol = np.where(m.party == "D", "#2166ac", np.where(m.party == "R", "#b2182b", "#555"))
                fig, ax = plt.subplots(figsize=(15, 9.5))
                ax.plot(*np.vstack([vt, vt[0]]).T, "k-", lw=1)
                ax.scatter(*P.T, c=pcol, s=22, alpha=.65, edgecolor="white", lw=.3, zorder=3)
                boxes = np.array([[-.62, .25], [1.62, .25], [.5, 1.42]])           # left, right, above
                for k in range(3):
                    img = cloud(top_words(results[f"V{k+1}_vs_rest"][0], +1), COLORS[3][k], 700, 420)
                    ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=.5), vt[k], xybox=boxes[k], boxcoords="data",
                                                 frameon=True, bboxprops=dict(ec=COLORS[3][k], lw=1.5),
                                                 arrowprops=dict(arrowstyle="-", color=COLORS[3][k], lw=1.2)))
                    ax.annotate(vertex_label(m, 3, k), boxes[k], xytext=(0, 128), textcoords="offset points",
                                ha="center", va="bottom", fontsize=10, color=COLORS[3][k], fontweight="bold")
                ax.scatter([], [], c="#2166ac", label="Democrat"); ax.scatter([], [], c="#b2182b", label="Republican")
                ax.legend(frameon=False, loc="lower center", ncol=2)
                ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.15, 2.15); ax.set_ylim(-.15, 1.95)
                ax.set_title(f"{CONGRESS}th House, K=3, tau={tau:g}, {kind}: membership simplex with each community's "
                             f"distinctive words (vs the rest)", fontsize=11)
                plt.tight_layout(); plt.savefig(OUT / f"simplex_clouds_{tag}.png", dpi=130); plt.close()

        # tetrahedron (K=4): regular tetrahedron seen almost from above (oblique projection), so the four
        # corners form a quadrilateral: Democratic communities on the left, Republican ones on the right.
        # Front edges solid, the back edge (V1-V3) dashed. Members coloured by party, clouds at the corners.
        if K == 4:
            pi = m[[f"pi_{k+1}" for k in range(4)]].values
            vt3 = np.array([[-1, 1, -1], [-1, -1, 1], [1, -1, -1], [1, 1, 1]], float)    # V1..V4 corners
            proj = lambda X: X[:, :2] + X[:, [2]] * np.array([.25, .15])
            vt = proj(vt3)
            P = proj(pi @ vt3) + np.random.default_rng(0).normal(scale=.006, size=(len(pi), 2))
            pcol = np.where(m.party == "D", "#2166ac", np.where(m.party == "R", "#b2182b", "#555"))
            # tetrahedron large in the centre, one cloud in each figure corner, joined to its vertex by a line
            fig = plt.figure(figsize=(19, 11.5))
            ax = fig.add_axes([.27, .07, .46, .83])
            for a, b in combinations(range(4), 2):
                back = vt3[a, 2] < 0 and vt3[b, 2] < 0
                ax.plot(*vt[[a, b]].T, "k--" if back else "k-", lw=.9 if back else 1.1, alpha=.5 if back else 1)
            ax.scatter(*P.T, c=pcol, s=26, alpha=.65, edgecolor="white", lw=.3, zorder=3)
            ax.set_aspect("equal"); ax.axis("off")
            pad = .12
            ax.set_xlim(vt[:, 0].min() - pad, vt[:, 0].max() + pad); ax.set_ylim(vt[:, 1].min() - pad, vt[:, 1].max() + pad)
            # V1 top-left, V2 bottom-left, V3 bottom-right, V4 top-right (matches the projected corners)
            slots = {0: ([.01, .50, .25, .36], 1), 1: ([.01, .05, .25, .36], 1),
                     2: ([.74, .05, .25, .36], 0), 3: ([.74, .50, .25, .36], 0)}
            for k in range(4):
                rect, edge = slots[k]
                cax = fig.add_axes(rect)
                cax.imshow(cloud(top_words(results[f"V{k+1}_vs_rest"][0], +1), COLORS[4][k], 900, 560))
                cax.set_xticks([]); cax.set_yticks([])
                for s in cax.spines.values():
                    s.set_edgecolor(COLORS[4][k]); s.set_linewidth(1.5)
                cax.set_title(vertex_label(m, 4, k), fontsize=10, color=COLORS[4][k], fontweight="bold")
                fig.add_artist(ConnectionPatch(xyA=(edge, .5), coordsA="axes fraction", axesA=cax,
                                               xyB=vt[k], coordsB="data", axesB=ax,
                                               color=COLORS[4][k], lw=1.3))
            ax.scatter([], [], c="#2166ac", label="Democrat"); ax.scatter([], [], c="#b2182b", label="Republican")
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=2)
            fig.suptitle(f"{CONGRESS}th House, K=4, tau={tau:g}, {kind}: membership tetrahedron with each community's "
                         f"distinctive words (vs the rest); dashed = back edge", fontsize=12)
            plt.savefig(OUT / f"simplex_clouds_{tag}.png", dpi=120); plt.close()

print(f"Wrote outputs to {OUT}")
