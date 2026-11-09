# Project webpage

`index.html` is the finished, self-contained demo page (data inlined; Plotly and wordcloud2 load from cdnjs).
Open it in a browser or upload it as-is to the course site.

| File | What it is |
|---|---|
| `template.html` | Page source (HTML/CSS/JS) with a `__DATA__` placeholder |
| `build_data.py` | Computes all page data and writes `data.json` + `index.html` |
| `inputs/` | K=4, tau=1 memberships and community word tables from the `eunyoung121-patch-1` branch (`community_words.py` kept for reference) |
| `caucuses_118.csv` | bioguide → caucus (Freedom Caucus, Blue Dogs, Progressive Caucus, Squad, Problem Solvers, New Democrats, Main Street) |
| `committees_118.csv`, `committee_rosters_raw.csv`, `build_committees.py`, `fetch_committees.py`, `wiki_committees/` | bioguide → House committee (21 full committees), from Wikipedia 118th-Congress rosters (2024 snapshots; members with under a year on a committee are left out) |
| `caucus_rosters_raw.csv`, `build_caucuses.py` | Scraped 118th-Congress rosters (Wikipedia oldid / Wayback snapshots, Aug–Sep 2024) and the name+state matching script |

Rebuild (from the repo root; needs the raw speeches at `crec_parsed/parsed/crec_118.parquet`, which is git-ignored and must never be committed):

    uv run --with pyarrow python website/build_data.py

Member word lists use the same pipeline as `community_words.py` (floor speeches, ≥ 250 chars, before 2024-11-01; words used by
≥ 10 members; median-length rescaling) and score each member against the rest of the House with Fightin' Words.
Only word statistics end up in the page, never speech text.

The home-page filter (caucus / nearest community / committee) uses the same Fightin' Words pipeline for each group against all other members; member surnames and, for committees, the committee name are removed from group and member word lists.

Extra caucuses (Republican Study Committee, Republican Governance Group, Western Caucus, Congressional Black Caucus, Congressional Hispanic Caucus, Equality Caucus) are in `caucuses_extra_118.csv` (raw rosters: `caucus_rosters_extra_raw.csv`, script: `build_caucuses_extra.py`). The Tuesday Group is the same organization as the Republican Governance Group (renamed in 2020), so it appears once.
The Caucuses page was removed; caucus filtering lives on the home page.

The Links tab points to the GitHub repo and to `report.pdf`. Put the final report at `website/report.pdf` (next to `index.html`) before submitting, or change that link in `template.html` and rebuild.
