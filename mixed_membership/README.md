# Mixed-membership communities of House members (118th Congress)

Each member gets a membership vector pi_i on the simplex (weights on K communities), estimated with
Mixed-SCORE (STAT 236, Lectures 9-11) from the vote-agreement network plus speech embeddings.

Run from the repo root: `uv run python mixed_membership/run_118.py`

| File | What it is |
|---|---|
| `common.py` | Loads votes, builds the agreement matrix S, loads speech PCs |
| `mixed_score.py` | SCORE, covariates, vertex hunting, barycentric coordinates |
| `run_118.py` | Fits K=3 and K=4, makes the two plots below |
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
4. **Choosing K.** Scree plot of S (eigengap rationale, Lecture 9): lambda_2 = 75.0, lambda_3 = 7.0,
   lambda_4 = 2.7, lambda_5 = 0.6. K=2 (the parties) is clear; K=3 is a weak elbow; K=4 is not supported by the
   scree plot and is shown for exploration only.

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
- The vertex members depend on the vote cutoff. With a cutoff of 40 votes, members who served briefly
  (for example Rulli, Fong) became vertices.
- Speech features exist only for the 115th, 116th and 118th, so only the 118th is shown here.
- Words used at each vertex are not analyzed yet; that needs the raw Congressional Record speeches.
