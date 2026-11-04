STAT 236 GROUP PROJECT - WHAT WE HAVE DONE SO FAR
==================================================

GOAL
----
Predict how well a House incumbent does in their election, relative to what was
expected, using the legislative graph (members x bills: Yea/Nay votes,
sponsorship, cosponsorship) and the text of the bills.

Everything runs with the project's uv environment, from the repo root:
    uv run python <script>.py        (or open the notebooks with the .venv kernel)


1. LOADING THE DATA (the legislative graph)
-------------------------------------------
Script: build_graph.py   (run: uv run python build_graph.py --congress 118)
  Sources
    - Voteview bulk CSVs (no key): members, roll calls, individual votes.
    - Congress.gov API (needs CONGRESS_API_KEY in env or .env): which bills
      became law, bill sponsors. Responses are cached in data/raw/.
    - Bill text -> data/<congress>/text/<bill>.txt
  Graph definition
    - Nodes: House members, and every bill/joint resolution with a recorded
      House FINAL-PASSAGE roll call.
    - Edges: member voted Yea on that bill (edges.csv holds Yea only).
    - Bill nodes carry outcome (enacted / passed_house_not_enacted / failed)
      and yea/nay counts. Sponsors are saved separately in sponsors.csv.
  Output per Congress in data/<c>/: graph.graphml, bills.csv, edges.csv,
    sponsors.csv, text/, voteview/ (raw votes incl. Nay and absences).
  Congresses built: 115, 116, 117, 118, 119 (119th has no election results yet).
  Note: Yea/Nay/absent are rebuilt as a signed matrix from voteview votes:
    cast_code 1-3 = Yea (+1), 4-6 = Nay (-1), everything else = 0 (no relation).

Script: fetch_bill_info.py
  For each bill, pulls title, policy area, and the CRS summary (HTML stripped)
  from the Congress.gov API -> data/<c>/bill_info.csv. Done for 115-118.
  (build_graph.py had one fix: api_get now writes ASCII-escaped JSON to the
  cache, to avoid a Windows cp1252 UnicodeEncodeError.)


2. PROCESSED ELECTION RESULTS (the target)
------------------------------------------
Raw input (downloaded manually from MIT Election Lab, "U.S. House 1976-2024"):
    data/elections/1976-2024-house.tab  + codebook-us-house-1976-2024.md
Script: build_targets.py  ->  data/elections/targets.csv
  Cleaning
    - Keep general elections only (no specials, runoffs, write-ins).
    - Drop EXHAUSTED/BLANK/SCATTER-type rows; sum fusion-ticket rows.
    - File has \"" inside quoted names; replaced with ' before parsing.
  Matching members -> results
    - By NAME WITHIN STATE (district numbers are unreliable after redistricting).
    - Surname must equal a run of the candidate's name tokens; accepted if there
      is first/middle/nickname evidence OR same district + party.
  Each Congress is paired with the election held at the END of its term:
    election year for Congress N = 2018 + 2(N-115)  (115->2018, 116->2020,
    117->2022, 118->2024). Features come from the Congress; the target is how
    that Congress's members then did.
  Uncontested seats: vote_share = margin = 1.0, two_party_share = NaN.
  TARGET: "tilt" = how much better/worse than expected a candidate did:
      tilt = (own two-party share_t - own two-party share_{t-1})
             - (own PARTY's national two-party share_t - same, t-1)
    i.e. change in the candidate's D/(D+R) share, minus the national swing
    (uniform-swing version; national share computed over contested D-vs-R
    districts only). National D share: 2016 .487, 2018 .519, 2020 .506,
    2022 .494, 2024 .489.
  tilt_c = tilt minus its (congress, party) mean. This is an optimistic,
    oracle-style target (it removes the year/party effect using test-year info).
  Mean tilt by Congress: 115 -0.021, 116 +0.003, 117 +0.005, 118 -0.001.
  Columns: congress, election_year, bioguide, name, state, district, party, ran,
    cand, res_district, vote_share, margin, won, uncontested, two_party_share,
    prev_* versions, national_share, prev_national_share, tilt.


3. EVALUATION PROTOCOL (same everywhere)
----------------------------------------
  - Train on Congresses 115 + 116, TEST on 118.
  - 117 is set aside (redistricting after the 2020 census breaks the
    previous-share baseline).
  - Hyperparameters (ridge penalty, XGBoost depth/trees) chosen by
    leave-one-Congress-out on 115/116.
  - Metric: out-of-sample R^2 against the training mean, on tilt and tilt_c.
  - Caveat: only ONE test year (~320 members, ~160 per party), so differences
    of ~0.01 R^2 are noise.


4. FEATURES AND MODELS WE TRIED
-------------------------------
All notebooks live in the repo root unless noted. Model families: ridge
(numpy), random forest, XGBoost (shallow, regularized), pooled vs separate
Democratic/Republican models.

(a) tilt_features.ipynb - graph features from the Yea graph
    20 features: n_yeas, share of Yeas on failed bills, mean bill popularity,
    own/other-party support, off-party share, crossover Yeas, SVD ideology
    (2 dims), centroid distances, co-Yea cosine strength, eigenvector
    centrality, PageRank, kNN betweenness, kNN cross-party share, n_sponsored.
    Result: prev_share is the only reliable predictor (negative coefficient =
    regression to the mean). Graph features add ~nothing.

(b) Disloyalty (in bill_embeddings.ipynb, reused afterwards)
    Votes against own party on roll calls where the party was >=75/25 one way.
    Features: n_cross (log1p count), cross_rate. Replaces "loyalty" because it
    avoids the absence problem. Adds ~+0.01 R^2 on top of prev_share
    (0.077 -> 0.087 pooled tilt_c), stable negative sign, mostly Democrats.

(c) bill_embeddings.ipynb - text embeddings
    - Model: BAAI/bge-small-en-v1.5 (local sentence-transformer, 384-d, CLS
      pooling, normalized) on bill title + summary (cached data/<c>/bill_emb.npy).
    - Member vector = vote-weighted mean of CENTERED bill embeddings
      (Yea=+1, Nay=-1, absent=0), optionally minus the party mean
      ("party-adjusted"). PCA fit on training Congresses only.
    - Tried: ridge on PCs, XGBoost on PCs, issue-area profiles, pooled vs
      party-separate. Result: nothing stable beyond prev_share + disloyalty.

(d) spectral_embeddings.ipynb - graph embeddings (SVD, not node2vec)
    - Signed SVD of the member x bill vote matrix (columns centered), k = 2,
      3, 5, 10, rotation-invariant features, benchmarked against DW-NOMINATE.
    - Ran with and without disloyalty, with ridge and XGBoost (section 8).
    - Result: no gain over prev_share (+ disloyalty).

(e) cosponsorship/  (set aside as a curiosity; see below)
    Result: weighting barely matters; adds at most ~+0.01 pooled; no feature
    has a stable sign across 2018/2020/2024; stacking more features hurts.

SUMMARY OF FINDINGS
  pooled tilt_c out-of-sample R^2:
      prev_share only .................... 0.077
      + disloyalty ....................... 0.087
      + ideology / embeddings / cosponsorship: no reliable improvement
  Previous share (regression to the mean) is the only robust predictor.
  Disloyalty is the only extra feature with a consistent sign.


5. THE COSPONSORSHIP GRAPH (side project, folder: cosponsorship/)
-----------------------------------------------------------------
  cosponsorship/build_cosponsor.py
    Downloads GovInfo BILLSTATUS bulk XML for all House bills/resolutions
    (hr, hres, hjres, hconres) for 115-119; raw zips go to
    cosponsorship/data/raw/ (git-ignored, rebuildable).
    Run: uv run python -I cosponsorship/build_cosponsor.py --congress 115 116 117 118 119
  Outputs in cosponsorship/data/<c>/: members.csv, bills.csv, sponsorship.csv
    (bipartite bill-member list with dates), edges.csv (member-member), and
    graph.graphml.
  Edge: two members are linked if they both sponsored/cosponsored the same bill
    (withdrawn cosponsors excluded). Three weightings:
      raw count | norm = sum 1/(k-1) | idf = sum log(N_members / k)
    (k = number of participants on the bill; discounts popular resolutions.)
    graph.graphml `weight` = idf, `count` = raw, `weight_norm` = 1/(k-1).
  cosponsorship/cosponsor_features.ipynb
    Rebuilds the projection using only cosponsorships dated before Nov 1 of the
    election year (no lame-duck leakage); features: strength, cross-party
    share, eigenvector centrality, PageRank, betweenness, kNN cross share, plus
    behaviour features (n_sponsored, n_cosponsored, niche-bill share,
    cross-party cosponsor rate, bipartisan draw rate).
  Caveats: the 119th data is partial; delegates are in the graph; the 1/(k-1)
    weighting is numerically unstable in party-separate models on raw tilt.


6. OTHER FILES
--------------
  visualize_graph.ipynb ... visualization of the vote graph (pre-existing)
  pyproject.toml, uv.lock ... environment (scikit-learn, xgboost,
                              sentence-transformers added)
  .gitignore ................ ignores .venv, data/raw, data/*/voteview,
                              data/*/text, cosponsorship/data/raw


7. KNOWN LIMITATIONS / IDEAS NOT YET DONE
-----------------------------------------
  - Single test year; tilt_c uses oracle (congress, party) means.
  - Lame-duck votes are still in the vote-based features (cosponsorship
    features are date-filtered).
  - Not yet tried: district presidential vote (lean against party, ticket
    splitting, redistricting flag), tenure / absence-rate features, more
    Congresses (pre-115), node2vec, a partisan-mix cosponsorship weight,
    community / bridge analysis of the cosponsorship graph.
