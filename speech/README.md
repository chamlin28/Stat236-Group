# Congressional Record speech pipeline (112th to 119th Congresses)

Scripts that download, parse, embed, and evaluate floor speeches from the Congressional Record.
The large data files are not stored in this repository (size limits). They are shared separately:

**Data download (Google Drive): [PASTE LINK HERE]**

Contents of the Drive folder: `crec_parsed.zip` (one Parquet file per Congress, 112 to 119) and
`crec_embeddings.zip` (speech and member embeddings for the 115th, 116th, and 118th, with metadata).

## Small outputs kept in this repository
- `data/speech/speech_features.csv`: one row per member per Congress (115, 116, 118) with speech features.
- `data/speech/speech_model_results.csv`: model comparison table.

## Data
Source: daily edition of the Congressional Record (GovInfo), January 2011 to October 29, 2026
(3,126 days, about 924,640 speeches). Download and parsing steps adapt the boundaries_congress package
(van der Does and Kolchinsky, 2021) and the congressional-record parser it builds on.

Parquet columns: `speech`, `chamber` (H or S), `is_extension`, `date`, `speaker`, `speaker_bioguide`
(member ID, empty for presiding officers and similar roles), `vol`, `num`, `congress_num`, `pages`,
`doc_title`, `title`. The index is `speech_id`.

Read in Python: `pd.read_parquet("parsed/crec_118.parquet")`.
Read in R: `arrow::open_dataset("parsed")`.

## Pipeline order
1. `download_crec.py <congress>` then `parse_crec.py <congress>` (batch version: `run_slurm.sh`).
2. `speech_embeddings.py <congress>` (batch version: `run_embeddings.sh`).
3. `speech_features.py` builds member-level features from the saved embeddings.
4. `run_speech_models.py` builds the features and compares models in one command.

Set `CREC_DATA_DIR` to the data folder. The `.sh` files contain cluster paths that must be edited.
Helper scripts: `compare_congresses.py`, `inspect_crec.py`, `join_check.py`, `pc_inspect.py`, `check_ids.py`, `check_env.py`.

## Speech selection for the features
House speeches with a member ID, at least 250 characters, dated before November 1 of the election year.
Embedding model: BAAI/bge-small-en-v1.5. Long speeches are split into chunks of up to 300 words (at most six
per speech); chunk embeddings are averaged by word count, then averaged to the member level.
Principal components are fit on the 115th and 116th only, then applied to the 118th, and party means are
subtracted within each Congress.

## Result
Training on the 115th and 116th and testing on the 118th (320 incumbents), speech features did not reliably
improve on party and previous two-party vote share. On the centered tilt target the baseline has an
out-of-sample R2 of 0.077; the overall PCs 1-3 set has 0.071 (gain in mean squared error -0.00000,
95% interval [-0.00002, 0.00001]). See `data/speech/speech_model_results.csv` for all feature sets.

## Known limitations
- The first-day (January 3) Records of several Congresses are missing because GovInfo has no package for them.
- The 119th Congress is incomplete (data end October 29, 2026).
- The daily edition differs slightly from the bound edition.
- About a third of rows have no member ID (mostly presiding officers).
- The House median speech length in the 119th differs sharply from the 118th; the cause has not been investigated.
- Only one test election (2024) is used.

## References
Zhao, M. Y. (2025). Words Speak As Loudly As Actions: Deep Learning Methods for Stance-Based Ideal Points
from Congressional Speeches. Bachelor's thesis, Harvard University.
van der Does, T. and Kolchinsky, A. (2021). boundaries_congress (GitHub: tamaravdd/boundaries_congress).
