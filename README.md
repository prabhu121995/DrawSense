# DrawSense

Ranks Singapore Pools TOTO numbers with machine learning and shows the **next 5 ranked six-number sets**.
You add each new draw result as it comes out; training happens in a Jupyter notebook that picks the best algorithm.

```
Add results (Save draw) ──> toto_results.csv ──> notebook.ipynb (retrain, pick best) ──> next 5 ranked sets
                            (appended)            artifacts/best_model/                   (shown right away)
```

> TOTO draws are random. Every six-number set has the same 1 in 13,983,816 chance of winning Group 1, and every
> number a 6/49 = 12.245% chance. The notebook tests honestly whether any model beats random selection on draws
> it never saw. On the current data none does, so treat the ranked sets as model estimates, not better odds.

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
3. retrains by re-running `notebook.ipynb` (about 2–3 minutes);
4. shows the next 5 ranked sets for the following draw.

Untick "Retrain and predict…" to only save. You can also upload a CSV/XLSX: only dates not stored yet are added.
**Fix a mistake** deletes the latest draw. Before the first change, a one-time copy of the original file is
saved as `toto_results.backup.csv`.

**Next sets tab:** the next draw date (Monday/Thursday), the 5 ranked sets (1–10 configurable), all 49 numbers
with calibrated probabilities, and a CSV download. A note appears when draws were added after the model was
trained: those draws are already used as history, and retraining makes the model learn from them too.

**Model tab:** the chosen algorithm, how it did on unseen test draws vs random, all algorithms' test results,
and a **Retrain now** button that re-runs `notebook.ipynb` on all stored draws (about 3 minutes).

## Training notebook

`notebook.ipynb` (run it in VS Code/Jupyter with the `.venv` kernel, or via the Retrain button):
1. loads `toto_results.csv` (8 columns only) and validates it;
2. builds features for each number from earlier draws only (frequencies over 10/20/50/100 draws, weighted
   recency, gaps since last seen, pair co-occurrence, odd/even, ranges, previous-draw context);
3. benchmarks 11 algorithms: random, frequency, weighted frequency, Logistic Regression, Random Forest, Extra
   Trees, XGBoost, LightGBM, CatBoost, MLP, and an ensemble. Validation is chronological walk-forward
   (retrain every 10 draws, predict the next ones before seeing them). The last 20% of draws are a final test
   that is never used for training or selection;
4. picks the best algorithm by the lowest log loss on the validation period, then reports its test result
   vs random with a significance test;
5. exports `artifacts/best_model/` (`model.joblib`, `metadata.json`, `leaderboard.csv`, `selection.csv`,
   `history.csv`). The app reloads it automatically.

Headless: `uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb`

## How the 5 sets are ranked

1. All 49 numbers are ranked by calibrated probability (the model's estimate that the number is one of the
   six main numbers).
2. Every 6-number combination of the top 12 numbers is scored by the average probability of its numbers.
3. Sets are picked best-first. A set may share at most 3 numbers with any higher-ranked set. Set #1 is always
   the six best-ranked numbers.

## Command line

```powershell
uv run toto-ml predict                                    # print the next 5 ranked sets
uv run toto-ml add 2026-11-19 3 8 17 25 40 49 --additional 11   # add a draw result
uv run toto-ml api                                        # optional JSON API, docs at http://127.0.0.1:8000/docs
```

API endpoints: `GET /api/model`, `POST /api/predict`, `GET /api/draws`, `POST /api/draws`.

## Tests

```powershell
uv run pytest                        # 45 tests: validation, leakage, evaluation, store, artifact, API
```

## Project layout

```
app.py                      DrawSense Streamlit app
notebook.ipynb              training, algorithm selection, artifact export
toto_results.csv            your results file: training data, new draws are appended here
configs/default.yaml        seed, feature windows, split, model hyper-parameters
src/toto_ml/
  store.py                  add / merge / delete draws in toto_results.csv
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
```

## Current result (313 draws, 20 Nov 2023 → 16 Nov 2026)

Best on validation: **CatBoost**. On the 63 unseen test draws it averaged 1.27 hits in its top 10, against
1.2245 for random selection (95% CI 1.06–1.51). The Holm-adjusted p-value is 1.00: no statistically
significant evidence that it beats random selection, which is the expected result for a fair lottery.
