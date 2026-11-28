# DrawSense

Singapore Pools **TOTO and 4D** in one SQLite database (`drawsense.db`), with Axis & Mirror metrics for every
drawn number, and a mobile-first app. For TOTO it also ranks numbers with machine learning and shows **4 ranked six-number sets**
for the next draw: 2 from the train/test split model and 2 from the model trained on 100% of draws.

```
once:        notebook.ipynb ──> benchmarks 11 algorithms, exports the XGBoost + Random Forest ensemble ──> artifacts/best_model/
every draw:  Save draw ──> toto_results.csv ──> quick refit of the 100% data model (seconds) ──> 4 sets
```

The full benchmark runs once (or occasionally). After that DrawSense predicts from the saved model. When you
add a new draw, it only refits the chosen model on all draws, which takes a few seconds.

> TOTO draws are random. Every six-number set has the same 1 in 13,983,816 chance of winning Group 1, and every
> number a 6/49 = 12.245% chance. The notebook tests honestly whether any model beats random selection on draws
> it never saw. On the current data none does, so treat the ranked sets as model estimates, not better odds.

## One database for TOTO and 4D

```
Get latest results ──> toto_scraper.py  ──> toto_results.csv  ─┐
                   └─> fourd_scraper.py ──> fourd_results.csv ─┴─> drawsense.db ──> TOTO model + next move
```

* Both scrapers are incremental: they only fetch draws after the last one in their CSV. The first 4D download
  (about 470 draws since Nov 2023) takes about 10 minutes; after that each run takes seconds.
* The CSV files stay the source of truth. Each ETL run (and each change in the app) rebuilds that game's rows in
  `drawsense.db`, so the database always matches the files. `uv run toto-ml db` rebuilds it by hand.
* The game is a **filter**: every table has a `game` column (`'toto'` or `'4d'`), or use the per-game tables.
  The prediction algorithm is TOTO-only (6 of 49) and never reads 4D data; 4D is stored and analysed separately.

| Table | Contents |
|---|---|
| `games` | `toto`, `4d` |
| `draws` | one row per draw of either game: `game`, `draw_number`, `draw_date` (ISO) |
| `toto_results` | the TOTO CSV, typed: n1..n6, additional, Group 1 prize, Group 1-7 share amount / winning shares |
| `fourd_results` | the 4D CSV: first, second, third, starter_1..10, consolation_1..10 (4-digit text, `0018` keeps its zeros) |
| `draw_numbers` | every drawn number of both games, one row each, with `digit_sum`, `root`, `axis`, `mirror` (4D also `front` / `back` pairs) |
| `number_metrics` | the Axis & Mirror reference chart, 00–99 (`in_chart = 1` for the document's 01–89) |
| `v_draw_number_metrics` | `draw_numbers` plus the chart metrics of each 4D number's front and back pair |
| `etl_runs` | when each ETL ran, per game, and what it added |

**Axis & Mirror metrics** (from the "Axis & Mirror Numbers Reference Chart (01 - 89)"), applied digit by digit,
so they work for TOTO's 2 digits and 4D's 4 digits:

* **root**: digit sum reduced to one digit (27 → 9, 55 → 1, 8064 → 9)
* **axis**: each digit swapped with its axis partner 0↔5, 1↔6, 2↔7, 3↔8, 4↔9 (27 → 72, 8064 → 3519)
* **mirror**: digits reversed (14 → 41, 8064 → 4608)

```sql
-- every 1st prize with root 9
SELECT draw_date, number, axis, mirror FROM draw_numbers WHERE game = '4d' AND prize = '1st' AND root = 9;
-- TOTO numbers whose mirror was drawn in the same draw
SELECT a.draw_date, a.number, b.number AS mirror_drawn FROM draw_numbers a
JOIN draw_numbers b ON b.draw_id = a.draw_id AND b.number = a.mirror AND a.number < b.number WHERE a.game = 'toto';
```

## The app (mobile first)

The app uses the full screen width on phones and desktops. At the top: both games' latest draw, one
**Get latest results** button for both, and a **TOTO | 4D** switch (`?game=4d` in the URL opens 4D directly).

* **TOTO**: your next move, Number guide, **Axis & Mirror** (latest draw with R/A/M, number check, all numbers
  with metrics + CSV), Add results, History, Model.
* **4D**: pick any draw → 1st/2nd/3rd, starter and consolation prizes with root/axis/mirror under each number;
  **Number check** (how often the number, its axis, its mirror and any permutation were drawn), **History**
  (filter by prize, period, root, digits; CSV), **Stats** (root spread, repeated numbers, back pairs, axis/mirror
  follow-ups), the **Axis & Mirror chart**.
* **Database** panel at the bottom: rows per table, ETL log, download `drawsense.db`, rebuild from the CSV files.

## Your next move, prize value and Get latest results

* **Your next move** is shown first. It is saved in `artifacts/next_move.csv` (+ `.json`) whenever the model is
  updated, so the app shows it instantly instead of recomputing on every visit. It is rebuilt only when
  `toto_results.csv` has a draw the saved move does not know yet.
* **Get latest results** (top of the app, or `uv run toto-ml etl`) runs `toto_scraper.py` and `fourd_scraper.py`
  (only draws after the last one saved), stores both games in `drawsense.db`, adds TOTO draws to `toto_results.csv`, replaces any hand-typed row with the official one, updates the
  full-history model and saves the new next move.
* **Prize value**: Groups 1-4 split their prize among all winners. Using the official prize data in
  `toto_results.csv`, DrawSense learns how popular each number is with players (`src/toto_ml/value.py`) and builds
  sets from the model's top 18 numbers that fewer players choose, in a typical draw shape (sum, odd/even, spread).
  On the current data, draws whose numbers were the least popular quarter paid a median Group 3 prize of $1,961,
  against $1,317 for the most popular quarter (+49%).

## Setup (Windows)

```powershell
winget install astral-sh.uv          # once
cd E:\Client\toto\toto_scraper
uv sync                              # installs everything into .venv
uv run streamlit run app.py          # opens DrawSense at http://localhost:8501
```

`uv run toto-ml ui` starts the same app.

## Using DrawSense

**Add results tab:** this is the main flow. Enter the draw date (pre-filled with the next draw), the six
winning numbers and the additional number, then click **Save draw**. DrawSense:
1. checks the draw (6 different numbers in 1–49, additional not one of them, new date);
2. appends it to `toto_results.csv` in the file's own format (`Thu, 19 Nov 2026`; prize columns blank);
3. updates the model: refits the 100% data ensemble on all draws with the same settings (a few seconds, no
   re-benchmark). The split model is not changed;
4. shows the 4 ranked sets for the following draw.

Untick "Update the model…" to only save. You can also upload a CSV/XLSX: only dates not stored yet are added,
then the model is updated. **Fix a mistake** deletes the latest draw. Before the first change, a one-time copy
of the original file is saved as `toto_results.backup.csv`.

**Next sets tab:** the next draw date (Monday/Thursday), the 4 ranked sets (how many from each model is configurable), all 49
numbers with calibrated probabilities, and a CSV download.

**Past matches tab:** answers "are draws taken from old draws?". Pick six numbers (the latest draw by default,
or a ticket) to list every stored draw that shares 3+ of them, next to how many such matches pure chance gives.
Below it, the whole history is compared with 500 simulated random histories of the same length.

**Model tab:** the chosen algorithm, how it did on unseen test draws vs random, and all algorithms' test results.
- **Update model** is only enabled when `toto_results.csv` has draws the model has not learned yet.
- **Full benchmark (occasional)** re-runs `notebook.ipynb` to re-test all 11 algorithms and pick the best again
  (about 3 minutes). It runs in the background with a progress bar at the top of the page, and refreshing the
  page doesn't stop it. Run it only when you want to re-check the algorithm choice, for example after many new
  draws. The quick update keeps the test evidence and calibration from the last full benchmark.

## Training notebook

`notebook.ipynb` (run it once in VS Code/Jupyter with the `.venv` kernel, or via **Full benchmark**):
1. loads `toto_results.csv` (8 columns only) and validates it;
2. builds features for each number from earlier draws only (frequencies over 10/20/50/100 draws, weighted
   recency, gaps since last seen, pair co-occurrence, odd/even, ranges, previous-draw context);
3. benchmarks 11 algorithms: random, frequency, weighted frequency, Logistic Regression, Random Forest, Extra
   Trees, XGBoost, LightGBM, CatBoost, MLP, and an ensemble. Validation is chronological walk-forward
   (retrain every 10 draws, predict the next ones before seeing them). The last 20% of draws are a final test
   that is never used for training or selection;
4. exports the model set in `configs/default.yaml` (`live.model: ensemble`, members XGBoost + Random Forest).
   The automatic pick (lowest validation log loss) is still shown for reference. The ensemble's final-test
   result is reported vs random with a significance test;
5. exports `artifacts/best_model/` (`model.joblib`, `metadata.json`, `leaderboard.csv`, `selection.csv`,
   `history.csv`). The app reloads it automatically.

Headless: `uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`

## Where the 4 sets come from

Both models are the same ensemble: XGBoost and Random Forest, combined by averaging each number's rank.

| Sets | Model | Trained on |
|---|---|---|
| #1, #2 | train/test split model | only the draws before the final test period: the exact model the test evidence is about. Fixed until the next full benchmark |
| #3, #4 | 100% data model | every draw. Refitted each time you save a draw |

Sets #3 and #4 may share at most 3 numbers with sets #1 and #2, so they are never near-copies. Change the split
with `live.sets_split` / `live.sets_full` in `configs/default.yaml`.

## How sets are ranked (within each model)

1. All 49 numbers are ranked by calibrated probability (the model's estimate that the number is one of the
   six main numbers).
2. Every 6-number combination of the top 12 numbers is scored by the average probability of its numbers.
3. Sets are picked best-first. A set may share at most 3 numbers with any higher-ranked set. Each model's
   first set is its six best-ranked numbers.

## Command line

```powershell
uv run toto-ml predict                                    # print the 4 sets (--split 2 --full 2)
uv run toto-ml add 2026-11-19 3 8 17 25 40 49 --additional 11   # add a draw result
uv run toto-ml etl                                        # latest TOTO + 4D -> CSV files -> drawsense.db -> model
uv run toto-ml etl --no-4d                                # TOTO only
uv run toto-ml db                                         # rebuild drawsense.db from the two CSV files
uv run python fourd_scraper.py --full                     # re-check the whole 4D list and fill gaps
uv run toto-ml api                                        # optional JSON API, docs at http://127.0.0.1:8000/docs
```

API endpoints: `GET /api/model`, `POST /api/predict` (form fields `n_split`, `n_full`), `GET /api/draws`, `POST /api/draws`.

## Tests

```powershell
uv run pytest                        # tests: validation, leakage, evaluation, store, artifact, API, repeats
```

## Project layout

```
app.py                      DrawSense Streamlit app
notebook.ipynb              training, algorithm selection, artifact export
toto_results.csv            your TOTO results file: training data, new draws are appended here
fourd_results.csv           4D results (fourd_scraper.py)
drawsense.db                SQLite: both games + Axis & Mirror metrics (rebuilt from the CSV files, not in git)
configs/default.yaml        seed, feature windows, split, model hyper-parameters
src/toto_ml/
  store.py                  add / merge / delete draws in toto_results.csv
  db.py                     drawsense.db: schema, CSV -> database sync, queries for the app
  metrics.py                Axis & Mirror: root, axis, mirror, reference chart
  data/                     CSV/XLSX loading and validation
  features/engineer.py      per-number features from past draws only
  models/                   baselines, ML models, ensemble, calibration, Optuna tuning
  evaluation/               walk-forward backtest, metrics, significance tests
  prediction/sets.py        number ranking and ranked sets
  artifact.py               select best, save/load artifact, predict next sets
  api/main.py               optional JSON API
  cli.py                    toto-ml ui / predict / add / api
tests/
toto_scraper.py             scraper that produces toto_results.csv (see README.txt)
fourd_scraper.py            scraper that produces fourd_results.csv
```

## Current result (315 draws, 20 Nov 2023 → 23 Nov 2026)

Live model: **Ensemble (XGBoost + Random Forest)**. On the 63 unseen test draws it averaged 1.30 hits in its
top 10, against 1.2245 for random selection (95% CI 1.11–1.49). The Holm-adjusted p-value is 1.00: no
statistically significant evidence that it beats random selection, which is the expected result for a fair
lottery. (The automatic rule would have picked CatBoost; its lead was also within noise.)
