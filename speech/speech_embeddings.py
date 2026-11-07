"""Embed House speeches and build member-level mean embeddings for one Congress.

Usage:
  python speech_embeddings.py 118 --pilot 3000 --graph-repo ~/Stat236-Group   # small test
  python speech_embeddings.py 118 --graph-repo ~/Stat236-Group                # full run
  Add --dummy to test the pipeline without loading the real model.

Defaults: House rows with a member ID, at least 250 characters, dated before Nov 1 of
the election year (--include-post keeps later speeches). Long speeches are split into
300-word chunks (at most 6 per speech), and chunk embeddings are averaged, weighted by words.
Model: BAAI/bge-small-en-v1.5 (384-d, normalized), the same one used for the bill embeddings.

Outputs in $CREC_DATA_DIR/embeddings/ (prefix pilot_ in pilot mode):
  speech_emb_<N>.npy      one row per kept speech (float32, unit length)
  speech_meta_<N>.parquet date, bioguide, n_chars, n_words, is_extension (same row order)
  member_emb_<N>.npy      length-weighted mean embedding per member
  member_meta_<N>.csv     bioguide, n_speeches, n_words, share_extension (same row order)
"""
import argparse, logging, os, sys, time, zlib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("congress", type=int)
ap.add_argument("--min-chars", type=int, default=250)
ap.add_argument("--include-post", action="store_true")
ap.add_argument("--pilot", type=int, default=0, help="embed a random sample of this many speeches")
ap.add_argument("--chunk-words", type=int, default=300)
ap.add_argument("--max-chunks", type=int, default=6)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--device", default=None)
ap.add_argument("--dummy", action="store_true")
ap.add_argument("--graph-repo", default=os.path.expanduser("~/Stat236-Group"))
args = ap.parse_args()

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
c = args.congress
DIM = 384
election_year = 2 * c + 1788                   # 115 -> 2018, 118 -> 2024
cutoff = pd.Timestamp(f"{election_year}-11-01")

base = os.environ.get("CREC_DATA_DIR", os.path.expanduser("~/crec_data"))
out_dir = os.path.join(base, "embeddings")
os.makedirs(out_dir, exist_ok=True)
prefix = "pilot_" if args.pilot else ""

# ---- 1. Select speeches ----
d = pd.read_parquet(os.path.join(base, "parsed", f"crec_{c}.parquet"),
                    columns=["date", "chamber", "speaker_bioguide", "speech", "is_extension"])
logging.info("Congress %d: %d rows", c, len(d))
d = d[(d.chamber == "H") & d.speaker_bioguide.notna()]
logging.info("House rows with a member ID: %d", len(d))
d = d[d.speech.str.len() >= args.min_chars]
logging.info("At least %d characters: %d", args.min_chars, len(d))
if not args.include_post:
    d = d[d.date < cutoff]
    logging.info("Before %s: %d", cutoff.date(), len(d))
if args.pilot:
    d = d.sample(min(args.pilot, len(d)), random_state=0).sort_index()
    logging.info("Pilot sample: %d speeches", len(d))

words = d.speech.str.split()
nwords = words.str.len().values
keep = nwords >= 1
d, words, nwords = d[keep], words[keep], nwords[keep]

# ---- 2. Chunk long speeches ----
chunks, owner, cw = [], [], []
for i, w in enumerate(words):
    for k, s in enumerate(range(0, len(w), args.chunk_words)):
        if k >= args.max_chunks:
            break
        piece = w[s:s + args.chunk_words]
        chunks.append(" ".join(piece)); owner.append(i); cw.append(len(piece))
logging.info("%d speeches -> %d chunks", len(d), len(chunks))

# ---- 3. Embed ----
if args.dummy:
    def encode(texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for r, t in enumerate(texts):
            for tok in t.lower().split():
                h = zlib.crc32(tok.encode())
                out[r, h % DIM] += 1.0 if (h >> 9) & 1 else -1.0
        nrm = np.linalg.norm(out, axis=1, keepdims=True); nrm[nrm == 0] = 1
        return out / nrm
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=args.device)
    model.max_seq_length = 512
    def encode(texts):
        return model.encode(texts, batch_size=args.batch_size,
                            normalize_embeddings=True, show_progress_bar=False)

order = np.argsort([len(t) for t in chunks])     # similar lengths together, less padding
E = np.zeros((len(chunks), DIM), dtype=np.float32)
B, t0 = 4096, time.time()
for b in range(0, len(order), B):
    idx = order[b:b + B]
    E[idx] = encode([chunks[j] for j in idx])
    done = min(b + B, len(order))
    logging.info("embedded %d / %d chunks (%.1f min elapsed)", done, len(order), (time.time() - t0) / 60)

# ---- 4. Chunks -> speeches (word-weighted mean) ----
owner = np.array(owner); cw = np.array(cw, dtype=np.float32)
starts = np.r_[0, np.flatnonzero(np.diff(owner)) + 1]
S = np.add.reduceat(E * cw[:, None], starts, axis=0)
S /= np.linalg.norm(S, axis=1, keepdims=True)
assert len(S) == len(d)

meta = pd.DataFrame({"date": d.date.values, "bioguide": d.speaker_bioguide.values,
                     "n_chars": d.speech.str.len().values, "n_words": nwords,
                     "is_extension": d.is_extension.values}, index=d.index)
np.save(os.path.join(out_dir, f"{prefix}speech_emb_{c}.npy"), S.astype(np.float32))
meta.to_parquet(os.path.join(out_dir, f"{prefix}speech_meta_{c}.parquet"))

# ---- 5. Member-level means ----
w = meta.n_words.values.astype(np.float32)
rows, M = [], []
for b, idx in meta.groupby("bioguide").indices.items():
    v = (S[idx] * w[idx, None]).sum(0); v /= np.linalg.norm(v)
    M.append(v)
    rows.append({"bioguide": b, "n_speeches": len(idx), "n_words": int(w[idx].sum()),
                 "share_extension": float(meta.is_extension.values[idx].mean())})
M = np.array(M, dtype=np.float32); mm = pd.DataFrame(rows)
np.save(os.path.join(out_dir, f"{prefix}member_emb_{c}.npy"), M)
mm.to_csv(os.path.join(out_dir, f"{prefix}member_meta_{c}.csv"), index=False)
logging.info("Saved %d speech embeddings and %d member embeddings to %s", len(S), len(M), out_dir)

# ---- 6. Is the main variation just party? ----
edges_path = os.path.join(args.graph_repo, "data", str(c), "edges.csv")
if os.path.exists(edges_path):
    party = pd.read_csv(edges_path).drop_duplicates("member").set_index("member").party
    min_n = 2 if args.pilot else 5
    ok = mm.bioguide.isin(party.index).values & (mm.n_speeches.values >= min_n)
    ok &= mm.bioguide.map(party).isin(["D", "R"]).values
    X = M[ok]; y = (mm.bioguide[ok].map(party) == "D").values.astype(float)
    print(f"\nPARTY CHECK on {ok.sum()} members with >= {min_n} speeches ({int(y.sum())} D, {int((1 - y).sum())} R)")
    def pcs(X, label):
        X = X - X.mean(0)
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        var = s ** 2 / (s ** 2).sum()
        sc = U * s
        cors = [abs(np.corrcoef(sc[:, k], y)[0, 1]) for k in range(3)]
        print(f"{label}: variance explained by PC1-5 = {[round(float(x), 3) for x in var[:5]]}")
        print(f"{label}: |correlation with party| for PC1-3 = {[round(float(x), 2) for x in cors]}")
    pcs(X, "raw member means   ")
    Xa = X.copy()
    for g in (0.0, 1.0):
        Xa[y == g] -= Xa[y == g].mean(0)
    pcs(Xa, "party-adjusted     ")
