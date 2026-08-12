# UMAR Residential Load Forecasting

This is a terminal-based Python forecasting system for the UMAR Residential
Load Dataset. It produces 96 predictions (one per 15-minute interval) using:

- a Probability Density Function method based on Kernel Density Estimation;
- Random Forest Regression; or
- both methods together.

## Dataset inspection and unit decision

The supplied workbook was inspected before the forecasting code was built.
The time-series data is in the `LoadData` worksheet:

| Column | Use |
| --- | --- |
| `DateTime` | Primary timestamp (rounded to the intended 15-minute boundary) |
| `Date`, `Time` | Human-readable checks for the timestamp |
| `DayOfWeek`, `Weekend` | Calendar checks; the program derives these again |
| `Demand_kW` | Forecast target: average electrical power during the interval |
| `Energy_kWh` | Energy consumed during that 15-minute interval |
| `Valid_samples` | Number of valid one-minute meter samples in the interval |

The values are **power in kW**, not energy. The workbook metadata explicitly
states this, and the numeric cross-check confirms:

```text
Energy_kWh = Demand_kW × 0.25 hours
```

Across all non-missing pairs, the maximum difference from this equation is
0.00005 kWh, which is explained by the four-decimal rounding in the workbook.

Key findings:

- period: 2019-07-01 through 2023-06-30;
- 140,256 rows: 1,461 days × 96 intervals;
- no duplicate timestamps and no missing timestamp slots;
- 1,458 missing `Demand_kW`/`Energy_kWh` values, all with `Valid_samples = 0`;
- 2,008 intervals have fewer than 15 valid one-minute samples;
- no negative load values;
- maximum measured demand is 4.0374 kW;
- high positive readings are flagged for review but retained because they are
  plausible residential peaks and agree with the energy column.

Several missing runs last more than a day. Therefore, the cleaner does not draw
a straight interpolation line across them. It uses only earlier information:
the same slot one day/week earlier, expanding similar-day means, then a
past-only fallback.

## Installation

Python 3.10 or newer is recommended. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`openpyxl` is required by pandas to read the supplied `.xlsx` file.

## Run the program

Interactive mode (the workbook is auto-detected when it is the only dataset in
the directory):

```powershell
python energy_forecast.py
```

Non-interactive example:

```powershell
python energy_forecast.py --date 01/07/2024 --method both
```

Useful options:

```powershell
# Inspect and clean the data without training
python energy_forecast.py --inspect-only

# Rebuild the saved model
python energy_forecast.py --retrain --date 01/07/2024 --method both

# Save a PNG comparison chart as well as the CSV
python energy_forecast.py --date 01/07/2024 --method both --plot
```

The first forecasting run tunes and evaluates the models, records the locked
test metrics, then refits both production RF models on all available data. The
models are saved to `artifacts/random_forest_model.joblib`. Later runs
reuse it unless the source dataset or program version changes. Forecast CSV
files are written to `predictions/`.

## Time split and leakage prevention

The supplied dataset uses this chronological split:

- training: 2019-07-01 to 2021-12-31;
- validation: 2022-01-01 to 2022-12-31;
- test: 2023-01-01 to 2023-06-30.

No random train-test split is used. Five Random Forest configurations are
trained only on 2019-2021 and compared on 2022. The configurations vary:

- number of trees;
- maximum tree depth;
- minimum samples per leaf; and
- features considered by each tree.

The configuration with the lowest 2022 validation RMSE is selected. The
program then follows two separate stages:

1. **Locked evaluation model:** refit on 2019-2022 and evaluated once on 2023.
   No 2023 target or 2023 historical-profile value enters this model. The
   printed validation/test metrics come only from this honest evaluation.
2. **Production forecast models:** after the test metrics are recorded, refit
   the selected direct and rolling settings on every measured row from July
   2019 through June 2023. The direct model and all-history profiles are used
   for a distant date; the rolling production model is used only when the
   necessary prior measurements exist.

The production model has no new unseen test set; its quality estimate is the
previously recorded locked 2023 result. This is the standard "evaluate first,
then refit on all available history" workflow.

## Two Random Forest modes

The program now keeps two deliberately different Random Forest models:

1. **Direct future-date RF:** for a distant date such as July 2024. It uses only
   calendar fields and historical profiles that are genuinely known in
   advance. It cannot use real June 2024 lag values because they do not exist.
2. **Rolling-lag RF:** for the locked 2023 historical comparison, or a true
   near-term situation where recent meter readings are available. It adds the
   previous 15-minute, previous-day, previous-week, one-hour rolling, and
   24-hour rolling loads. Every one of these inputs is shifted, so the current
   target and future measurements are never used.

Both modes compare an unweighted baseline with recency-weighted candidates on
2022. The current validation selected no recency weighting for the direct RF,
but a 365-day half-life for the rolling RF. This means newer readings count
more for the rolling model while older seasonal examples are retained at lower
weight.

For a date inside the dataset, `--historical-model` controls which saved models
are compared with the actual values:

- `production` is the default and uses both all-data RF models. It is useful to
  inspect model fit, but the date may have been part of training, so this is not
  an honest future-accuracy result.
- `locked` uses both through-2022 RF models and through-2022 PDF history for an
  honest January-June 2023 test comparison.

The terminal and CSV show:

- actual measured load;
- whether each RF result is from the all-data production model or the locked
  through-2022 model;
- direct and rolling RF predictions;
- the PDF/KDE prediction using history selected by the same scope; and
- interval and daily errors.

All-data production comparison:

```powershell
python energy_forecast.py --date 29/06/2023 --method both --historical-model production --plot
```

Honest locked comparison:

```powershell
python energy_forecast.py --date 29/06/2023 --method both --historical-model locked --plot
```

Verified locked-2023 performance after this change:

| Model | MAE (kW) | MAPE | R-squared |
| --- | ---: | ---: | ---: |
| Direct RF | 0.0812 | 41.22% | -0.0398 |
| Rolling-lag RF | 0.0523 | 24.39% | 0.3737 |

The rolling model reduced locked-test MAE by 35.52% and MAPE by 40.83%.
In locked mode for 29 June 2023, its daily-energy error is 8.67%; at 21:00 it predicts
0.2096 kW versus the measured 0.2180 kW, an absolute error of 0.0084 kW.

Missing target rows are excluded from metric calculations. MAPE also excludes
true zero-load rows because division by zero is undefined.

`create_features()` creates the requested calendar, lag, rolling-average, and
similar-day features. Every historical aggregate is shifted so a row can only
use earlier observations. The explicit historical averages are:

- average at the same 15-minute time across permitted history;
- average for the same weekday and time;
- average for weekday/weekend type and time;
- average for the same season, weekday, and time; and
- similar-day average for the same month, weekday, and time.

The production Random Forest is a **direct long-horizon model**. It uses
calendar features and leakage-safe historical profile features that are known
for any future date. Raw 15-minute/day/week lag columns are created for
short-horizon experiments, but deliberately excluded from this model: for a
date such as 1 July 2024, the actual loads on 30 June 2024 do not exist in this
dataset. Substituting them during a backtest would make the accuracy appear
better than a real future forecast.

Feature availability for 1 July 2024:

| Feature | Available? | Treatment |
| --- | --- | --- |
| Year/month/day/weekday/hour/minute/slot | Yes | Used directly |
| Weekend and season | Yes | Used directly |
| Historical same-time/weekday/similar-day averages | Yes | Calculated from all available history through June 2023 |
| Actual previous 15-minute load | No | Created for short-horizon studies, excluded from this direct model |
| Actual previous-day load | No | Excluded because 30 June 2024 is not measured |
| Actual previous-week load | No | Excluded because 24 June 2024 is not measured |
| Rolling load average immediately before the date | No | Excluded because its required 2024 loads are unknown |

The PDF method selects historical observations with this fallback order:

1. same month, weekday, and 15-minute interval;
2. same season, weekday, and interval;
3. same month, weekday/weekend type, and interval;
4. same season, day type, and interval;
5. same day type and interval;
6. same interval.

For each interval it reports the KDE expected value, median, mode, and central
90% prediction limits. These limits describe historical variability; they are
not a guarantee about future occupancy or weather.

## Output interpretation

All load forecasts are in kW. Daily energy is calculated by summing:

```text
predicted load in kW × 0.25 hours
```

The terminal prints daily energy, average/minimum/maximum load, peak time, and
the complete 96-row table. With `both`, it also prints the daily and
interval-level difference between the methods. The CSV includes the detailed
PDF statistics, Random Forest values, interval energy, sample counts, and the
similar-day rule used.

These are statistical forecasts from historical meter patterns. They do not
include weather forecasts, occupancy schedules, appliance changes, or tariffs,
so long-horizon results should be interpreted as expected profiles rather than
precise operational commitments.
