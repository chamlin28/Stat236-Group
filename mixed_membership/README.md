# Mixed-membership communities of House members (118th Congress)

Each member gets a membership vector pi_i on the simplex (weights on K communities), estimated with
Mixed-SCORE (STAT 236, Lectures 9-11) from the vote-agreement network plus speech embeddings.

Run from the repo root: `uv run python mixed_membership/run_118.py`

| File | What it is |
|---|---|
| `common.py` | Loads votes, builds the agreement matrix S, loads speech PCs |
| `mixed_score.py` | SCORE, covariates, vertex hunting, barycentric coordinates |
| `run_118.py` | Fits K=3 and K=4, makes the two plots below |
| `plot_agreement_graph.py` | Draws the agreement network and the matrix S |
| `agreement_network_118.png` | The weighted agreement graph (network layout and heatmap of S) |
| `agreement_matrix_118.png` | The weighted agreement matrix S on its own |
| `plot_scree_118.py`, `scree_118.png` | Scree plot (absolute eigenvalues of S, linear and log scale) |
| `scree_speech_118.png` | Scree plot of the speech covariates (raw and standardized) |
| `simplex_K3_118.png` | 3-vertex model (triangle) |
| `simplex_K4_118.png` | 4-vertex model (tetrahedron, two views) |

## Method
1. **Network.** Final-passage House roll calls (360 bills). `S[i,j]` = share of bills both members voted on where
   they voted the same way (Yea/Nay). Used as a weighted network, no thresholding. Members need at least half
   of the votes (180), which keeps 428 of 445.
2. **Covariates.** Speech-embedding PCs 1-5 per member (`data/speech/speech_features.csv`), standardized.
3. **Mixed-SCORE with attributes (Lecture 10 and Ji & Jin in Lecture 11).** SCORE matrix R from the top K
   eigenvectors of S; H = top K left singular vectors of the covariates; embed rows of [R, tau*H] with tau = 1
   (covariate block scaled to the same total robust spread as the graph block); vertex hunting by k-means
   denoising then SPA (SVS*-style); barycentric coordinates by constrained least squares; pi proportional to w / b_1.
4. **Choosing K.** Scree plot of S (eigengap rationale, Lecture 9; `scree_118.png`): the top eigenvalues are
   316.8, 75.0, 7.0, 2.7. K=2 (the parties) is the clear gap. Lambda_3 and lambda_4 are also well above the noise
   bulk (S has a zero diagonal, which shifts every eigenvalue down by 1: the bulk sits near -1, and near 0, with
   maximum about 0.4, once the diagonal is set to 1). Lambda_5 and lambda_6 are only slightly above it. So K=3 and
   K=4 are both defensible; the scree plot does not single one out.

## Reading the vertices
Labels are from member names and NOMINATE scores; the repo has no caucus data. Nearest members are by pi.

| K | Vertex | D share | NOM1 | Nearest members | Reading |
|---|---|---|---|---|---|
| 3 | V1 | 74% | -0.17 | Pingree, Napolitano, Case, Larsen, Mullin | Mainstream Democrats |
| 3 | V2 | 37% | +0.16 | Rogers, Letlow, Womack, Hudson, Carter | Mainstream Republicans |
| 3 | V3 | 19% | +0.46 | Roy, Massie, Biggs, Rosendale, Crane | Freedom Caucus Republicans |
| 4 | V1 | 81% | -0.25 | Bush, Pressley, Ramirez, Tlaib, Lee | Far-left Democrats |
| 4 | V2 | 80% | -0.18 | Peltola, Levin, Vasquez, Peters, Panetta | Moderate Democrats |
| 4 | V3 | 21% | +0.30 | Lucas, Latta, LaTurner, Graves, Wagner | Mainstream Republicans |
| 4 | V4 | 14% | +0.52 | Roy, Biggs, Rosendale, Crane, Good | Freedom Caucus Republicans |

"D share" is the pi-weighted Democratic share. Few members sit at a vertex (about 8% have a weight above 0.9),
so most are mixtures.

## Caveats
- K=3 is a descriptive choice, not a tested one. The SgnQ test needs a binary graph and was not used.
- tau = 1 is our choice. The notes only call tau a tuning parameter. With the speech block weighted this
  heavily, vertex positions are sensitive to it.
- The speech block has no scree gap once standardized (`scree_speech_118.png`): the 5 PCs carry 26%, 22%, 20%,
  18%, 14% of the variance, so H (top K singular vectors) is K of 5 near-equally weighted directions. Unstandardized,
  PC1 carries 61%. We used the standardized version; K is not chosen separately for the speech block.
- The vertex members depend on the vote cutoff. With a cutoff of 40 votes, members who served briefly
  (for example Rulli, Fong) became vertices.
- Speech features exist only for the 115th, 116th and 118th, so only the 118th is shown here.

## Words by community (`community_words.py`)
Run after `run_118.py` (which now also saves `memberships_K{K}_tau{0,1}_118.csv`, with pi, NOMINATE and the
SCORE coordinates). Needs `crec_parsed/parsed/crec_118.parquet`, `pyarrow` and `wordcloud`; outputs go to `words/`.

- One document per member: 118th House floor speeches (no Extensions of Remarks, >= 250 characters, before
  2024-11-01). Unigrams + bigrams used by >= 10 members; procedural words and state names removed; each member's
  counts rescaled to the median length so every member counts once (373 members with >= 1000 words).
- Community counts are pi-weighted (soft membership). Distinctive words: Fightin' Words z-scores (Monroe,
  Colaresi & Quinn 2008), dropping words where one member supplies > 40% of a group's usage.
- Contrasts: each community vs the rest, and within each party its two largest communities (removes the plain
  party difference). `words_*.csv` has every word with |z| > 1.96.
- Uses the tau = 1 memberships. Caveat: those communities use speech PCs, so their words differ partly by
  construction.
- Two runs: words + two-word phrases (no suffix) and single words only (`_words` suffix). Both use the same
  members (single-word count >= 1000).
- Figures: `clouds_*` (one cloud per community), `clouds_within_party_*`, `eigenmap_K3_*` (members at the SCORE
  coordinates of S, coloured by pi, with each community's cloud at its vertex), `simplex_clouds_K3_*` (the
  membership triangle with each community's cloud at its corner), `simplex_clouds_K4_*` (the tetrahedron, drawn
  as a fixed 2D projection with the back edge V1-V3 dashed, a cloud at each of the four corners).
