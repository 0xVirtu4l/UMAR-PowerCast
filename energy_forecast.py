"""Terminal residential electricity-load forecasting for the UMAR dataset.

The program implements two forecasting methods:

1. A probability-density method based on Kernel Density Estimation (KDE).
2. A Random Forest regressor for direct, long-horizon forecasts.

Run ``python energy_forecast.py --help`` for command-line options.  With no
date or method arguments, the program asks for them interactively.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROGRAM_VERSION = "1.4.0"
INTERVAL_MINUTES = 15
INTERVAL_HOURS = INTERVAL_MINUTES / 60.0
INTERVALS_PER_DAY = 96

# These features are known for any requested future date.  Raw lags are also
# created by create_features(), but they are intentionally not used by the
# direct future model because the actual load one day before a far-future date
# is not known at forecast time.
RF_FEATURE_COLUMNS = [
    "year",
    "month",
    "day",
    "dayofweek",
    "hour",
    "minute",
    "quarter_hour",
    "is_weekend",
    "season",
    "minute_sin",
    "minute_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "historical_same_time_mean_kw",
    "historical_same_dow_mean_kw",
    "historical_similar_mean_kw",
    "historical_season_mean_kw",
    "historical_daytype_mean_kw",
]

# These extra features are available only when the immediately preceding real
# measurements exist.  They power the rolling one-step model used for the
# locked 2023 comparison, not the distant-date 2024 model.
ROLLING_RF_FEATURE_COLUMNS = RF_FEATURE_COLUMNS + [
    "lag_15min_kw",
    "lag_previous_day_kw",
    "lag_previous_week_kw",
    "rolling_mean_1h_kw",
    "rolling_mean_24h_kw",
]


@dataclass
class ColumnMapping:
    """Columns detected in the input file."""

    datetime: str | None
    date: str | None
    time: str | None
    load: str
    energy: str | None
    valid_samples: str | None


@dataclass
class DatasetReport:
    """A compact, printable record of dataset quality and cleaning actions."""

    source_file: str
    source_sheet: str | None
    raw_rows: int
    cleaned_rows: int
    start: str
    end: str
    unique_days: int
    rows_per_complete_day: int
    duplicate_timestamps_removed: int
    missing_grid_timestamps_inserted: int
    missing_load_values_imputed: int
    negative_or_nonfinite_values: int
    zero_load_values: int
    partial_meter_windows: int
    high_load_flags: int
    high_load_threshold_kw: float
    inferred_input_kind: str
    power_column: str
    energy_column: str | None
    energy_formula_max_error_kwh: float | None
    unit_explanation: str


def _normalise_name(value: Any) -> str:
    """Return a column name in a comparison-friendly form."""

    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _first_matching_column(
    columns: Sequence[Any], exact: Sequence[str], contains: Sequence[str] = ()
) -> str | None:
    normalised = {str(column): _normalise_name(column) for column in columns}
    for wanted in exact:
        wanted_normalised = _normalise_name(wanted)
        for original, simplified in normalised.items():
            if simplified == wanted_normalised:
                return original
    for wanted in contains:
        wanted_normalised = _normalise_name(wanted)
        for original, simplified in normalised.items():
            if wanted_normalised in simplified:
                return original
    return None


def detect_columns(columns: Sequence[Any]) -> ColumnMapping:
    """Detect timestamp, load, energy, and quality columns by their names."""

    datetime_col = _first_matching_column(
        columns,
        exact=("DateTime", "Timestamp", "Date_Time", "TimeStamp"),
        contains=("datetime", "timestamp"),
    )
    date_col = _first_matching_column(columns, exact=("Date", "ReadingDate"))
    time_col = _first_matching_column(columns, exact=("Time", "ReadingTime"))

    # Prefer an explicitly labelled power/demand column over an interval-energy
    # column when both exist, as in the UMAR workbook.
    load_col = _first_matching_column(
        columns,
        exact=(
            "Demand_kW",
            "Load_kW",
            "Power_kW",
            "Total_Active_Power",
            "Active_Power",
            "Demand",
            "Load",
            "Power",
        ),
        contains=("demandkw", "loadkw", "powerkw", "activepower"),
    )
    energy_col = _first_matching_column(
        columns,
        exact=("Energy_kWh", "Load_kWh", "Consumption_kWh", "Energy"),
        contains=("energykwh", "loadkwh", "consumptionkwh"),
    )

    # If no power column exists, an energy column can still be the forecasting
    # target; it will be converted to average kW during cleaning.
    if load_col is None and energy_col is not None:
        load_col = energy_col

    valid_col = _first_matching_column(
        columns,
        exact=("Valid_samples", "ValidSamples", "SampleCount", "QualityCount"),
        contains=("validsamples",),
    )

    if load_col is None:
        raise ValueError(
            "Could not detect a load column. Use a heading such as Demand_kW, "
            "Load_kW, Power_kW, Energy_kWh, or Consumption_kWh."
        )
    if datetime_col is None and (date_col is None or time_col is None):
        raise ValueError(
            "Could not detect a timestamp. Supply DateTime/Timestamp, or both "
            "Date and Time columns."
        )

    return ColumnMapping(
        datetime=datetime_col,
        date=date_col,
        time=time_col,
        load=load_col,
        energy=energy_col,
        valid_samples=valid_col,
    )


def _choose_excel_sheet(path: Path, requested_sheet: str | None) -> tuple[pd.DataFrame, str]:
    """Find the worksheet that contains timestamp and load columns."""

    if requested_sheet:
        return pd.read_excel(path, sheet_name=requested_sheet), requested_sheet

    workbook = pd.ExcelFile(path)
    errors: list[str] = []
    for sheet in workbook.sheet_names:
        sample = pd.read_excel(path, sheet_name=sheet, nrows=5)
        try:
            detect_columns(sample.columns)
        except ValueError as exc:
            errors.append(f"{sheet}: {exc}")
            continue
        return pd.read_excel(path, sheet_name=sheet), sheet

    raise ValueError(
        "No worksheet contains both a timestamp and a recognised load column. "
        f"Checked: {', '.join(workbook.sheet_names)}."
    )


def load_dataset(path: Path, sheet: str | None = None) -> tuple[pd.DataFrame, str | None]:
    """Read a CSV or Excel file and return its raw data and sheet name."""

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return _choose_excel_sheet(path, sheet)
    if suffix in {".csv", ".txt", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else None
        frame = pd.read_csv(path, sep=separator, engine="python")
        return frame, None
    raise ValueError("The data file must be CSV, TSV, XLS, XLSM, or XLSX.")


def _parse_datetime_values(values: pd.Series) -> pd.Series:
    """Parse normal datetimes or Excel serial-date numbers."""

    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors="coerce")

    numeric = pd.to_numeric(values, errors="coerce")
    numeric_fraction = float(numeric.notna().mean())
    if numeric_fraction > 0.9 and numeric.dropna().between(20_000, 80_000).mean() > 0.9:
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")

    parsed = pd.to_datetime(values, errors="coerce", dayfirst=True)
    if parsed.notna().mean() < 0.8:
        alternate = pd.to_datetime(values, errors="coerce", dayfirst=False)
        if alternate.notna().sum() > parsed.notna().sum():
            parsed = alternate
    return parsed


def _build_datetime(raw: pd.DataFrame, mapping: ColumnMapping) -> pd.Series:
    if mapping.datetime:
        timestamp = _parse_datetime_values(raw[mapping.datetime])
    else:
        combined = raw[mapping.date].astype(str).str.strip() + " " + raw[
            mapping.time
        ].astype(str).str.strip()
        timestamp = _parse_datetime_values(combined)

    # Excel stores dates as floating-point day counts.  Rounding removes tiny
    # artefacts such as 19:59:59.999997 while preserving the intended slot.
    return timestamp.dt.round(f"{INTERVAL_MINUTES}min")


def _infer_input_kind(
    raw: pd.DataFrame,
    mapping: ColumnMapping,
    unit_override: str,
) -> tuple[str, float | None]:
    """Determine whether the selected target contains kW or interval kWh."""

    if unit_override in {"kw", "kwh"}:
        return unit_override, None

    load_name = _normalise_name(mapping.load)
    if "kwh" in load_name or "energy" in load_name or "consumption" in load_name:
        return "kwh", None
    if "kw" in load_name or "power" in load_name or "demand" in load_name:
        formula_error: float | None = None
        if mapping.energy and mapping.energy != mapping.load:
            power = pd.to_numeric(raw[mapping.load], errors="coerce")
            energy = pd.to_numeric(raw[mapping.energy], errors="coerce")
            valid = power.notna() & energy.notna()
            if valid.any():
                formula_error = float((energy[valid] - power[valid] * INTERVAL_HOURS).abs().max())
        return "kw", formula_error

    # A relation to a separately labelled energy column is strong evidence
    # that the selected load column is power.
    if mapping.energy and mapping.energy != mapping.load:
        load = pd.to_numeric(raw[mapping.load], errors="coerce")
        energy = pd.to_numeric(raw[mapping.energy], errors="coerce")
        valid = load.notna() & energy.notna()
        if valid.any():
            error = float((energy[valid] - load[valid] * INTERVAL_HOURS).abs().median())
            scale = max(float(energy[valid].abs().median()), 1e-9)
            if error / scale < 0.02:
                return "kw", float(
                    (energy[valid] - load[valid] * INTERVAL_HOURS).abs().max()
                )

    raise ValueError(
        f"The unit of column {mapping.load!r} is ambiguous. Re-run with "
        "--unit kw if it is average power, or --unit kwh if it is energy per "
        "15-minute interval."
    )


def _causal_impute_load(frame: pd.DataFrame) -> pd.Series:
    """Fill missing loads using earlier values only.

    Long meter outages should not be bridged with straight-line interpolation.
    The preferred replacements are the same slot on a prior day/week, followed
    by expanding historical means and finally a past-only forward fill.
    """

    filled = frame["load_kw"].copy()

    # Repeated passes allow a multi-day gap to inherit an earlier known daily
    # or weekly profile.  All shifts are positive, so no future row is used.
    for _ in range(8):
        before = int(filled.isna().sum())
        filled = filled.fillna(filled.shift(INTERVALS_PER_DAY))
        filled = filled.fillna(filled.shift(7 * INTERVALS_PER_DAY))
        if int(filled.isna().sum()) == before:
            break

    timestamp = frame["datetime"]
    helper = pd.DataFrame(
        {
            "load": filled,
            "month": timestamp.dt.month,
            "dow": timestamp.dt.dayofweek,
            "weekend": (timestamp.dt.dayofweek >= 5).astype(int),
            "slot": timestamp.dt.hour * 4 + timestamp.dt.minute // 15 + 1,
        },
        index=frame.index,
    )

    exact_history = helper.groupby(["month", "dow", "slot"], sort=False)[
        "load"
    ].transform(lambda series: series.shift(1).expanding(min_periods=1).mean())
    filled = filled.fillna(exact_history)

    daytype_history = helper.assign(load=filled).groupby(
        ["weekend", "slot"], sort=False
    )["load"].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )
    filled = filled.fillna(daytype_history)

    global_past_mean = filled.shift(1).expanding(min_periods=1).mean()
    filled = filled.fillna(global_past_mean).ffill()

    # This fallback matters only if a file starts with missing readings before
    # any historical value exists.  The UMAR data does not require it.
    if filled.isna().any():
        filled = filled.fillna(float(filled.median()))
    if filled.isna().any():
        raise ValueError("The load column contains no usable numeric values.")
    return filled


def clean_dataset(
    raw: pd.DataFrame,
    source_path: Path,
    source_sheet: str | None,
    unit_override: str = "auto",
) -> tuple[pd.DataFrame, DatasetReport, ColumnMapping]:
    """Detect the schema, clean the time series, and standardise it to kW."""

    mapping = detect_columns(raw.columns)
    input_kind, formula_error = _infer_input_kind(raw, mapping, unit_override)
    timestamp = _build_datetime(raw, mapping)
    source_load = pd.to_numeric(raw[mapping.load], errors="coerce")

    if input_kind == "kw":
        load_kw = source_load
    else:
        load_kw = source_load / INTERVAL_HOURS

    if mapping.energy and mapping.energy != mapping.load:
        source_energy = pd.to_numeric(raw[mapping.energy], errors="coerce")
    elif input_kind == "kwh":
        source_energy = source_load
    else:
        source_energy = load_kw * INTERVAL_HOURS

    if mapping.valid_samples:
        valid_samples = pd.to_numeric(raw[mapping.valid_samples], errors="coerce")
    else:
        valid_samples = pd.Series(np.nan, index=raw.index)

    frame = pd.DataFrame(
        {
            "datetime": timestamp,
            "load_kw": load_kw,
            "source_energy_kwh": source_energy,
            "valid_samples": valid_samples,
        }
    )
    frame = frame.dropna(subset=["datetime"])

    # Missing values are reported and imputed separately.  This count is only
    # for present-but-invalid measurements such as negative or infinite load.
    invalid_mask = frame["load_kw"].notna() & (
        ~np.isfinite(frame["load_kw"]) | (frame["load_kw"] < 0)
    )
    negative_or_nonfinite = int(invalid_mask.sum())
    frame.loc[invalid_mask, "load_kw"] = np.nan

    frame = frame.sort_values("datetime")
    duplicate_count = int(frame["datetime"].duplicated(keep="last").sum())
    frame = frame.drop_duplicates("datetime", keep="last")

    if frame.empty:
        raise ValueError("No valid timestamps remain after parsing the dataset.")

    expected_index = pd.date_range(
        frame["datetime"].min(), frame["datetime"].max(), freq=f"{INTERVAL_MINUTES}min"
    )
    original_index = pd.DatetimeIndex(frame["datetime"])
    inserted_count = int(len(expected_index.difference(original_index)))
    frame = frame.set_index("datetime").reindex(expected_index).rename_axis("datetime")
    frame = frame.reset_index()

    original_missing = frame["load_kw"].isna()
    missing_count = int(original_missing.sum())
    zero_count = int(frame["load_kw"].eq(0).sum())
    if frame["valid_samples"].notna().any():
        partial_windows = int(frame["valid_samples"].lt(INTERVAL_MINUTES).sum())
    else:
        partial_windows = 0

    valid_load = frame.loc[frame["load_kw"].notna(), "load_kw"]
    q1, q3 = valid_load.quantile([0.25, 0.75])
    high_threshold = float(q3 + 3.0 * (q3 - q1))
    high_flags = int((valid_load > high_threshold).sum())

    frame["load_kw"] = _causal_impute_load(frame)
    frame["energy_kwh"] = frame["load_kw"] * INTERVAL_HOURS
    frame["is_imputed"] = original_missing.to_numpy()

    unit_explanation = (
        "The selected column is average power in kW. Each 15-minute interval's "
        "energy is power x 0.25 hours."
        if input_kind == "kw"
        else "The selected column is interval energy in kWh; average power was "
        "calculated as energy / 0.25 hours."
    )

    rows_per_day = frame.groupby(frame["datetime"].dt.date).size()
    complete_days = int((rows_per_day == INTERVALS_PER_DAY).sum())
    report = DatasetReport(
        source_file=str(source_path.resolve()),
        source_sheet=source_sheet,
        raw_rows=int(len(raw)),
        cleaned_rows=int(len(frame)),
        start=str(frame["datetime"].min()),
        end=str(frame["datetime"].max()),
        unique_days=int(frame["datetime"].dt.date.nunique()),
        rows_per_complete_day=complete_days,
        duplicate_timestamps_removed=duplicate_count,
        missing_grid_timestamps_inserted=inserted_count,
        missing_load_values_imputed=missing_count,
        negative_or_nonfinite_values=negative_or_nonfinite,
        zero_load_values=zero_count,
        partial_meter_windows=partial_windows,
        high_load_flags=high_flags,
        high_load_threshold_kw=high_threshold,
        inferred_input_kind=input_kind,
        power_column=mapping.load,
        energy_column=mapping.energy,
        energy_formula_max_error_kwh=formula_error,
        unit_explanation=unit_explanation,
    )
    return frame, report, mapping


def print_dataset_report(report: DatasetReport, mapping: ColumnMapping) -> None:
    """Print the schema and cleaning report before any model is trained."""

    print("\nDATASET INSPECTION")
    print("=" * 80)
    print(f"File:                 {report.source_file}")
    if report.source_sheet:
        print(f"Worksheet:            {report.source_sheet}")
    if mapping.datetime:
        print(f"Timestamp column:     {mapping.datetime}")
    else:
        print(f"Date/time columns:    {mapping.date} + {mapping.time}")
    print(f"Power/target column:  {report.power_column}")
    if report.energy_column:
        print(f"Energy check column:  {report.energy_column}")
    print(f"Period:               {report.start} to {report.end}")
    print(f"Rows:                 {report.cleaned_rows:,}")
    print(
        f"Days:                 {report.unique_days:,} "
        f"({report.rows_per_complete_day:,} with 96 slots)"
    )
    print(f"Duplicates removed:   {report.duplicate_timestamps_removed:,}")
    print(f"Grid rows inserted:   {report.missing_grid_timestamps_inserted:,}")
    print(f"Missing loads filled: {report.missing_load_values_imputed:,}")
    print(f"Meter windows <15:    {report.partial_meter_windows:,}")
    print(f"Negative/non-finite:  {report.negative_or_nonfinite_values:,}")
    print(f"Zero loads retained:  {report.zero_load_values:,}")
    print(
        f"High-load flags:      {report.high_load_flags:,} above "
        f"{report.high_load_threshold_kw:.4f} kW (retained)"
    )
    print(f"Units:                {report.unit_explanation}")
    if report.energy_formula_max_error_kwh is not None:
        print(
            "Energy cross-check:   max |Energy_kWh - Demand_kW x 0.25| = "
            f"{report.energy_formula_max_error_kwh:.6f} kWh"
        )


def _season_from_month(month: pd.Series) -> pd.Series:
    # 0=winter, 1=spring, 2=summer, 3=autumn
    return ((month % 12) // 3).astype(int)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, interval, weekend, season, and cyclical features."""

    result = frame.copy()
    dt = pd.to_datetime(result["datetime"])
    result["year"] = dt.dt.year
    result["month"] = dt.dt.month
    result["day"] = dt.dt.day
    result["dayofweek"] = dt.dt.dayofweek
    result["hour"] = dt.dt.hour
    result["minute"] = dt.dt.minute
    result["quarter_hour"] = dt.dt.hour * 4 + dt.dt.minute // 15 + 1
    result["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    result["season"] = _season_from_month(result["month"])

    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    result["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    result["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    result["dow_sin"] = np.sin(2 * np.pi * result["dayofweek"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["dayofweek"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12)
    day_of_year = dt.dt.dayofyear
    result["dayofyear_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    result["dayofyear_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return result


def _past_group_mean(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Expanding group mean shifted by one occurrence to prevent leakage."""

    return frame.groupby(keys, sort=False)["load_kw"].transform(
        lambda series: series.shift(1).expanding(min_periods=1).mean()
    )


def create_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Create model features using historical values only.

    The lag and rolling columns are useful for short-horizon experiments and
    diagnostics.  The direct Random Forest uses RF_FEATURE_COLUMNS so it can
    forecast an arbitrary future date without inventing unknown future lags.
    """

    features = add_calendar_features(cleaned)
    features["lag_15min_kw"] = features["load_kw"].shift(1)
    features["lag_previous_day_kw"] = features["load_kw"].shift(INTERVALS_PER_DAY)
    features["lag_previous_week_kw"] = features["load_kw"].shift(
        7 * INTERVALS_PER_DAY
    )
    past_load = features["load_kw"].shift(1)
    features["rolling_mean_1h_kw"] = past_load.rolling(4, min_periods=1).mean()
    features["rolling_mean_24h_kw"] = past_load.rolling(
        INTERVALS_PER_DAY, min_periods=4
    ).mean()

    features["historical_same_time_mean_kw"] = _past_group_mean(
        features, ["quarter_hour"]
    )
    features["historical_same_dow_mean_kw"] = _past_group_mean(
        features, ["dayofweek", "quarter_hour"]
    )
    features["historical_similar_mean_kw"] = _past_group_mean(
        features, ["month", "dayofweek", "quarter_hour"]
    )
    features["historical_season_mean_kw"] = _past_group_mean(
        features, ["season", "dayofweek", "quarter_hour"]
    )
    features["historical_daytype_mean_kw"] = _past_group_mean(
        features, ["is_weekend", "quarter_hour"]
    )

    # Early rows may not yet have an exact month/day group.  Every fallback is
    # an expanding statistic shifted by one row, so it uses only prior loads.
    features["historical_same_dow_mean_kw"] = features[
        "historical_same_dow_mean_kw"
    ].fillna(features["historical_same_time_mean_kw"])
    features["historical_daytype_mean_kw"] = features[
        "historical_daytype_mean_kw"
    ].fillna(features["historical_same_time_mean_kw"])
    features["historical_season_mean_kw"] = features[
        "historical_season_mean_kw"
    ].fillna(features["historical_same_dow_mean_kw"])
    features["historical_similar_mean_kw"] = features[
        "historical_similar_mean_kw"
    ].fillna(features["historical_season_mean_kw"])
    features["historical_similar_mean_kw"] = features[
        "historical_similar_mean_kw"
    ].fillna(features["historical_daytype_mean_kw"])
    return features


def build_profile_features(
    history: pd.DataFrame, requested_rows: pd.DataFrame
) -> pd.DataFrame:
    """Build direct-forecast features from an explicitly permitted history.

    ``requested_rows`` may contain real targets for validation/testing or only
    timestamps for a future forecast.  None of those target values are read
    while the historical profiles are calculated.
    """

    future = add_calendar_features(requested_rows.copy().reset_index(drop=True))
    history = add_calendar_features(history.copy())
    history = history.loc[~history["is_imputed"]].copy()
    if history.empty:
        raise ValueError("No measured history is available for profile features.")

    same_time_profile = history.groupby("quarter_hour", observed=True)[
        "load_kw"
    ].mean()
    same_dow_profile = history.groupby(
        ["dayofweek", "quarter_hour"], observed=True
    )["load_kw"].mean()
    daytype_profile = history.groupby(
        ["is_weekend", "quarter_hour"], observed=True
    )["load_kw"].mean()
    season_profile = history.groupby(
        ["season", "dayofweek", "quarter_hour"], observed=True
    )["load_kw"].mean()
    exact_profile = history.groupby(
        ["month", "dayofweek", "quarter_hour"], observed=True
    )["load_kw"].mean()

    global_mean = float(history["load_kw"].mean())

    def lookup(profile: pd.Series, keys: Iterable[tuple[int, ...]]) -> np.ndarray:
        values: list[float] = []
        for key in keys:
            normalised_key: Any = tuple(key)
            if len(normalised_key) == 1:
                normalised_key = normalised_key[0]
            values.append(profile.get(normalised_key, np.nan))
        return np.asarray(values, dtype=float)

    same_time_values = lookup(
        same_time_profile,
        future[["quarter_hour"]].itertuples(index=False, name=None),
    )
    same_time_values = np.where(
        np.isfinite(same_time_values), same_time_values, global_mean
    )

    same_dow_values = lookup(
        same_dow_profile,
        future[["dayofweek", "quarter_hour"]].itertuples(index=False, name=None),
    )
    same_dow_values = np.where(
        np.isfinite(same_dow_values), same_dow_values, same_time_values
    )

    daytype_values = lookup(
        daytype_profile,
        future[["is_weekend", "quarter_hour"]].itertuples(index=False, name=None),
    )
    daytype_values = np.where(
        np.isfinite(daytype_values), daytype_values, same_time_values
    )

    season_values = lookup(
        season_profile,
        future[["season", "dayofweek", "quarter_hour"]].itertuples(
            index=False, name=None
        ),
    )
    season_values = np.where(
        np.isfinite(season_values), season_values, same_dow_values
    )

    exact_values = lookup(
        exact_profile,
        future[["month", "dayofweek", "quarter_hour"]].itertuples(
            index=False, name=None
        ),
    )
    exact_values = np.where(np.isfinite(exact_values), exact_values, season_values)

    future["historical_same_time_mean_kw"] = same_time_values
    future["historical_same_dow_mean_kw"] = same_dow_values
    future["historical_daytype_mean_kw"] = daytype_values
    future["historical_season_mean_kw"] = season_values
    future["historical_similar_mean_kw"] = exact_values
    return future


def build_future_features(
    cleaned: pd.DataFrame, prediction_date: pd.Timestamp
) -> pd.DataFrame:
    """Build all 96 known-in-advance feature rows for a future date."""

    start = prediction_date.normalize()
    requested = pd.DataFrame(
        {
            "datetime": pd.date_range(
                start, periods=INTERVALS_PER_DAY, freq=f"{INTERVAL_MINUTES}min"
            )
        }
    )
    return build_profile_features(cleaned, requested)


def _prepare_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    fill_values: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    matrix = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    if fill_values is None:
        fill_values = matrix.median().fillna(0.0)
    matrix = matrix.fillna(fill_values).fillna(0.0)
    return matrix, fill_values


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    """Calculate MAE, RMSE, safe MAPE, and R-squared."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    nonzero = np.abs(actual) > 1e-6
    mape = (
        float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else math.nan
    )
    return {
        "MAE_kW": float(mean_absolute_error(actual, predicted)),
        "RMSE_kW": float(math.sqrt(mean_squared_error(actual, predicted))),
        "MAPE_pct": mape,
        "R2": float(r2_score(actual, predicted)),
        "Rows": int(len(actual)),
        "MAPE_zero_rows_skipped": int((~nonzero).sum()),
    }


class PDFEstimator:
    """KDE forecaster that selects historical values from similar days."""

    LEVELS = [
        ("same month + weekday", ["month", "dayofweek", "quarter_hour"]),
        ("same season + weekday", ["season", "dayofweek", "quarter_hour"]),
        ("same month + day type", ["month", "is_weekend", "quarter_hour"]),
        ("same season + day type", ["season", "is_weekend", "quarter_hour"]),
        ("same day type", ["is_weekend", "quarter_hour"]),
        ("same interval", ["quarter_hour"]),
    ]

    def __init__(self, min_samples: int = 8, confidence: float = 0.90) -> None:
        self.min_samples = min_samples
        self.confidence = confidence
        self.group_samples: list[tuple[str, list[str], dict[Any, np.ndarray]]] = []

    @staticmethod
    def _normalise_key(values: Sequence[Any]) -> Any:
        key = tuple(int(value) for value in values)
        return key[0] if len(key) == 1 else key

    def fit(self, history: pd.DataFrame) -> "PDFEstimator":
        prepared = add_calendar_features(history)
        if "is_imputed" in prepared:
            prepared = prepared.loc[~prepared["is_imputed"]]
        prepared = prepared.loc[prepared["load_kw"].notna()]
        self.group_samples = []
        for label, keys in self.LEVELS:
            grouped: dict[Any, np.ndarray] = {}
            for key, values in prepared.groupby(keys, observed=True)["load_kw"]:
                if not isinstance(key, tuple):
                    key = (key,)
                grouped[self._normalise_key(key)] = values.to_numpy(dtype=float)
            self.group_samples.append((label, keys, grouped))
        return self

    def _select_samples(self, row: pd.Series) -> tuple[np.ndarray, str]:
        fallback: tuple[np.ndarray, str] | None = None
        for label, keys, groups in self.group_samples:
            key = self._normalise_key([row[key_name] for key_name in keys])
            samples = groups.get(key)
            if samples is None or len(samples) == 0:
                continue
            fallback = (samples, label)
            if len(samples) >= self.min_samples:
                return samples, label
        if fallback is None:
            raise ValueError("No historical samples are available for PDF forecasting.")
        return fallback

    def predict_expected(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = add_calendar_features(frame)
        predictions = []
        for _, row in prepared.iterrows():
            samples, _ = self._select_samples(row)
            predictions.append(float(np.mean(samples)))
        return np.asarray(predictions)

    def _distribution_statistics(self, samples: np.ndarray) -> dict[str, float]:
        samples = np.asarray(samples, dtype=float)
        samples = samples[np.isfinite(samples)]
        if len(samples) == 0:
            raise ValueError("Cannot estimate a PDF from an empty sample.")

        alpha = (1.0 - self.confidence) / 2.0
        if len(samples) < 2 or np.std(samples) < 1e-10:
            value = max(0.0, float(samples[0]))
            return {
                "pdf_expected_kw": value,
                "pdf_median_kw": value,
                "pdf_mode_kw": value,
                "pdf_lower_kw": value,
                "pdf_upper_kw": value,
            }

        try:
            kde = gaussian_kde(samples)
            standard_deviation = float(np.std(samples, ddof=1))
            lower_grid = max(0.0, float(np.min(samples) - 3 * standard_deviation))
            upper_grid = float(np.max(samples) + 3 * standard_deviation)
            if upper_grid <= lower_grid:
                upper_grid = lower_grid + 1e-6
            grid = np.linspace(lower_grid, upper_grid, 800)
            density = kde(grid)
            area = float(trapezoid(density, grid))
            if not np.isfinite(area) or area <= 0:
                raise ValueError("KDE returned a zero or non-finite area.")
            density = density / area
            cdf = np.concatenate(
                ([0.0], cumulative_trapezoid(density, grid))
            )
            cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)
            expected = float(trapezoid(grid * density, grid))
            median = float(np.interp(0.5, cdf, grid))
            lower = float(np.interp(alpha, cdf, grid))
            upper = float(np.interp(1 - alpha, cdf, grid))
            mode = float(grid[int(np.argmax(density))])
        except (np.linalg.LinAlgError, ValueError):
            expected = float(np.mean(samples))
            median = float(np.median(samples))
            lower = float(np.quantile(samples, alpha))
            upper = float(np.quantile(samples, 1 - alpha))
            rounded = np.round(samples, 4)
            values, counts = np.unique(rounded, return_counts=True)
            mode = float(values[int(np.argmax(counts))])

        return {
            "pdf_expected_kw": max(0.0, expected),
            "pdf_median_kw": max(0.0, median),
            "pdf_mode_kw": max(0.0, mode),
            "pdf_lower_kw": max(0.0, lower),
            "pdf_upper_kw": max(0.0, upper),
        }

    def predict_day(self, prediction_date: pd.Timestamp) -> pd.DataFrame:
        day = pd.DataFrame(
            {
                "datetime": pd.date_range(
                    prediction_date.normalize(),
                    periods=INTERVALS_PER_DAY,
                    freq=f"{INTERVAL_MINUTES}min",
                )
            }
        )
        prepared = add_calendar_features(day)
        records: list[dict[str, Any]] = []
        for _, row in prepared.iterrows():
            samples, basis = self._select_samples(row)
            stats = self._distribution_statistics(samples)
            records.append(
                {
                    "datetime": row["datetime"],
                    **stats,
                    "pdf_sample_count": int(len(samples)),
                    "pdf_similarity_basis": basis,
                }
            )
        return pd.DataFrame(records)


def _fit_random_forest(
    training: pd.DataFrame,
    parameters: dict[str, Any],
    feature_columns: Sequence[str] = RF_FEATURE_COLUMNS,
) -> tuple[RandomForestRegressor, pd.Series]:
    training = training.loc[~training["is_imputed"]]
    matrix, fill_values = _prepare_matrix(training, feature_columns)
    model = RandomForestRegressor(
        n_estimators=int(parameters["n_estimators"]),
        max_depth=parameters["max_depth"],
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=parameters["max_features"],
        n_jobs=-1,
        random_state=42,
    )
    sample_weight = None
    half_life_days = parameters.get("recency_half_life_days")
    if half_life_days:
        age_days = (
            training["datetime"].max() - training["datetime"]
        ).dt.total_seconds() / 86_400
        # A one-half weight after each half-life lets recent consumption
        # patterns matter more without deleting older seasons completely.
        sample_weight = np.power(0.5, age_days / float(half_life_days))
    model.fit(
        matrix,
        training["load_kw"].to_numpy(),
        sample_weight=sample_weight,
    )
    return model, fill_values


def _evaluation_row(
    split: str,
    method: str,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    return {"Split": split, "Method": method, **calculate_metrics(actual, predicted)}


def _hyperparameter_candidates(maximum_trees: int) -> list[dict[str, Any]]:
    """Small, beginner-readable Random Forest search evaluated on 2022."""

    if maximum_trees < 20:
        raise ValueError("--n-estimators must be at least 20 for model tuning.")
    lower = max(20, maximum_trees // 2)
    middle = max(lower + 1, int(round(maximum_trees * 0.75)))
    return [
        {
            "n_estimators": lower,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "max_features": 0.7,
            "recency_half_life_days": None,
        },
        {
            "n_estimators": lower,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "max_features": 0.7,
            "recency_half_life_days": 365,
        },
        {
            "n_estimators": middle,
            "max_depth": 18,
            "min_samples_leaf": 3,
            "max_features": 0.8,
            "recency_half_life_days": 730,
        },
        {
            "n_estimators": maximum_trees,
            "max_depth": None,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "recency_half_life_days": None,
        },
        {
            "n_estimators": maximum_trees,
            "max_depth": 24,
            "min_samples_leaf": 5,
            "max_features": 1.0,
            "recency_half_life_days": 365,
        },
    ]


def _chronological_splits(features: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, str]:
    """Use the requested calendar splits, with a generic fallback."""

    dt = features["datetime"]
    train = dt < pd.Timestamp("2022-01-01")
    validation = (dt >= pd.Timestamp("2022-01-01")) & (dt < pd.Timestamp("2023-01-01"))
    test = dt >= pd.Timestamp("2023-01-01")
    if train.any() and validation.any() and test.any():
        description = (
            "Training: 2019-2021 (available data starts 2019-07-01); "
            "validation: 2022; test: 2023-01-01 through dataset end"
        )
        return train, validation, test, description

    # For another dataset, preserve time order with 70%/15%/15% row splits.
    n_rows = len(features)
    train_end = int(n_rows * 0.70)
    validation_end = int(n_rows * 0.85)
    positions = np.arange(n_rows)
    description = "Training/validation/test: chronological 70%/15%/15%"
    return (
        pd.Series(positions < train_end, index=features.index),
        pd.Series((positions >= train_end) & (positions < validation_end), index=features.index),
        pd.Series(positions >= validation_end, index=features.index),
        description,
    )


def train_and_evaluate(
    cleaned: pd.DataFrame,
    n_estimators: int,
) -> dict[str, Any]:
    """Evaluate honestly, then refit a separate production model on all data."""

    print("\nCreating leakage-safe features...")
    causal_features = create_features(cleaned)
    train_mask, validation_mask, test_mask, split_description = _chronological_splits(
        causal_features
    )

    train_history = cleaned.loc[train_mask].copy()
    validation_history = cleaned.loc[validation_mask].copy()
    test_history = cleaned.loc[test_mask].copy()

    # Training rows use only expanding, shifted statistics.  The entire 2022
    # validation block uses fixed profiles from 2019-2021, and the entire 2023
    # test block uses fixed profiles from 2019-2022.  Therefore no validation
    # or test target can leak into another row's direct-forecast features.
    train = causal_features.loc[train_mask].copy()
    validation = build_profile_features(train_history, validation_history)
    validation = validation.loc[~validation["is_imputed"]].copy()

    permitted_through_2022 = cleaned.loc[train_mask | validation_mask].copy()
    test = build_profile_features(permitted_through_2022, test_history)
    test = test.loc[~test["is_imputed"]].copy()

    if validation.empty or test.empty:
        raise ValueError("Validation or test split has no measured target rows.")

    evaluation_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    rolling_tuning_rows: list[dict[str, Any]] = []
    candidates = _hyperparameter_candidates(n_estimators)

    print("[1/5] Tuning the direct future-date RF on the 2022 validation period...")
    best_parameters: dict[str, Any] | None = None
    best_validation_metrics: dict[str, float | int] | None = None
    best_rmse = math.inf
    for candidate_number, parameters in enumerate(candidates, start=1):
        print(
            f"      candidate {candidate_number}/{len(candidates)}: "
            f"trees={parameters['n_estimators']}, depth={parameters['max_depth']}, "
            f"leaf={parameters['min_samples_leaf']}, "
            f"features={parameters['max_features']}, "
            f"recency half-life={parameters['recency_half_life_days']}"
        )
        candidate_model, candidate_fill = _fit_random_forest(train, parameters)
        validation_x, _ = _prepare_matrix(
            validation, RF_FEATURE_COLUMNS, candidate_fill
        )
        candidate_prediction = np.maximum(
            0.0, candidate_model.predict(validation_x)
        )
        candidate_metrics = calculate_metrics(
            validation["load_kw"].to_numpy(), candidate_prediction
        )
        tuning_rows.append(
            {
                "Candidate": candidate_number,
                "Trees": parameters["n_estimators"],
                "Max_depth": "None"
                if parameters["max_depth"] is None
                else parameters["max_depth"],
                "Min_leaf": parameters["min_samples_leaf"],
                "Max_features": parameters["max_features"],
                "Half_life_days": parameters["recency_half_life_days"]
                if parameters["recency_half_life_days"] is not None
                else "None",
                **candidate_metrics,
                "Selected": False,
            }
        )
        if float(candidate_metrics["RMSE_kW"]) < best_rmse:
            best_rmse = float(candidate_metrics["RMSE_kW"])
            best_parameters = parameters
            best_validation_metrics = candidate_metrics

    if best_parameters is None or best_validation_metrics is None:
        raise RuntimeError("Random Forest hyperparameter tuning did not produce a model.")
    selected_index = min(
        range(len(tuning_rows)), key=lambda index: float(tuning_rows[index]["RMSE_kW"])
    )
    tuning_rows[selected_index]["Selected"] = True

    evaluation_rows.append(
        {"Split": "Validation", "Method": "RF direct", **best_validation_metrics}
    )

    # A separate rolling model can use real earlier measurements.  All lag and
    # rolling columns were shifted before the split, so the target row itself
    # and every future row remain unavailable to its features.
    rolling_train = causal_features.loc[train_mask].copy()
    rolling_validation = causal_features.loc[
        validation_mask & ~causal_features["is_imputed"]
    ].copy()
    print("[2/5] Tuning the rolling-lag RF on the 2022 validation period...")
    best_rolling_parameters: dict[str, Any] | None = None
    best_rolling_validation_metrics: dict[str, float | int] | None = None
    best_rolling_rmse = math.inf
    for candidate_number, parameters in enumerate(candidates, start=1):
        print(
            f"      candidate {candidate_number}/{len(candidates)}: "
            f"trees={parameters['n_estimators']}, depth={parameters['max_depth']}, "
            f"leaf={parameters['min_samples_leaf']}, "
            f"features={parameters['max_features']}, "
            f"recency half-life={parameters['recency_half_life_days']}"
        )
        candidate_model, candidate_fill = _fit_random_forest(
            rolling_train, parameters, ROLLING_RF_FEATURE_COLUMNS
        )
        rolling_validation_x, _ = _prepare_matrix(
            rolling_validation, ROLLING_RF_FEATURE_COLUMNS, candidate_fill
        )
        candidate_prediction = np.maximum(
            0.0, candidate_model.predict(rolling_validation_x)
        )
        candidate_metrics = calculate_metrics(
            rolling_validation["load_kw"].to_numpy(), candidate_prediction
        )
        rolling_tuning_rows.append(
            {
                "Candidate": candidate_number,
                "Trees": parameters["n_estimators"],
                "Max_depth": "None"
                if parameters["max_depth"] is None
                else parameters["max_depth"],
                "Min_leaf": parameters["min_samples_leaf"],
                "Max_features": parameters["max_features"],
                "Half_life_days": parameters["recency_half_life_days"]
                if parameters["recency_half_life_days"] is not None
                else "None",
                **candidate_metrics,
                "Selected": False,
            }
        )
        if float(candidate_metrics["RMSE_kW"]) < best_rolling_rmse:
            best_rolling_rmse = float(candidate_metrics["RMSE_kW"])
            best_rolling_parameters = parameters
            best_rolling_validation_metrics = candidate_metrics

    if best_rolling_parameters is None or best_rolling_validation_metrics is None:
        raise RuntimeError("Rolling Random Forest tuning did not produce a model.")
    selected_rolling_index = min(
        range(len(rolling_tuning_rows)),
        key=lambda index: float(rolling_tuning_rows[index]["RMSE_kW"]),
    )
    rolling_tuning_rows[selected_rolling_index]["Selected"] = True
    evaluation_rows.append(
        {
            "Split": "Validation",
            "Method": "RF rolling",
            **best_rolling_validation_metrics,
        }
    )

    pdf_validation = PDFEstimator().fit(train_history)
    pdf_validation_prediction = pdf_validation.predict_expected(validation)
    evaluation_rows.append(
        _evaluation_row(
            "Validation", "PDF/KDE", validation["load_kw"], pdf_validation_prediction
        )
    )

    print("[3/5] Re-fitting both evaluation models through 2022...")
    validation_for_training = build_profile_features(train_history, validation_history)
    train_validation = pd.concat(
        [train, validation_for_training], ignore_index=True
    ).sort_values("datetime")
    test_model, test_fill = _fit_random_forest(train_validation, best_parameters)
    rolling_train_validation = causal_features.loc[
        train_mask | validation_mask
    ].copy()
    rolling_test_model, rolling_test_fill = _fit_random_forest(
        rolling_train_validation,
        best_rolling_parameters,
        ROLLING_RF_FEATURE_COLUMNS,
    )

    print("[4/5] Running the locked final evaluation on 2023...")
    test_x, _ = _prepare_matrix(test, RF_FEATURE_COLUMNS, test_fill)
    test_prediction = np.maximum(0.0, test_model.predict(test_x))
    evaluation_rows.append(
        _evaluation_row("Test", "RF direct", test["load_kw"], test_prediction)
    )

    rolling_test = causal_features.loc[
        test_mask & ~causal_features["is_imputed"]
    ].copy()
    rolling_test_x, _ = _prepare_matrix(
        rolling_test, ROLLING_RF_FEATURE_COLUMNS, rolling_test_fill
    )
    rolling_test_prediction = np.maximum(
        0.0, rolling_test_model.predict(rolling_test_x)
    )
    evaluation_rows.append(
        _evaluation_row(
            "Test",
            "RF rolling",
            rolling_test["load_kw"],
            rolling_test_prediction,
        )
    )

    pdf_test = PDFEstimator().fit(permitted_through_2022)
    pdf_test_prediction = pdf_test.predict_expected(test)
    evaluation_rows.append(
        _evaluation_row("Test", "PDF/KDE", test["load_kw"], pdf_test_prediction)
    )

    # Only after the locked test predictions and metrics have been recorded do
    # we refit the production model.  Its features remain causal (all profile
    # averages are shifted), but it can now learn from every measured row.
    print("[5/6] Re-fitting the production direct RF on all data through June 2023...")
    production_model, production_fill = _fit_random_forest(
        causal_features, best_parameters
    )
    print("[6/6] Re-fitting the production rolling RF on all data through June 2023...")
    rolling_production_model, rolling_production_fill = _fit_random_forest(
        causal_features,
        best_rolling_parameters,
        ROLLING_RF_FEATURE_COLUMNS,
    )

    evaluation = pd.DataFrame(evaluation_rows)
    return {
        # The stored metrics above came from test_model, which never saw 2023.
        # The model saved for future forecasts is the later all-history refit.
        "model": production_model,
        "direct_evaluation_model": test_model,
        "direct_evaluation_fill_values": test_fill.to_dict(),
        "feature_columns": RF_FEATURE_COLUMNS,
        "fill_values": production_fill.to_dict(),
        "evaluation": evaluation.to_dict(orient="records"),
        "validation_tuning": tuning_rows,
        "rolling_validation_tuning": rolling_tuning_rows,
        "selected_hyperparameters": best_parameters,
        "selected_rolling_hyperparameters": best_rolling_parameters,
        "rolling_evaluation_model": rolling_test_model,
        "rolling_feature_columns": ROLLING_RF_FEATURE_COLUMNS,
        "rolling_fill_values": rolling_test_fill.to_dict(),
        "rolling_production_model": rolling_production_model,
        "rolling_production_fill_values": rolling_production_fill.to_dict(),
        "split_description": split_description,
        "training_mode": "locked_evaluation_then_refit_all",
        "evaluation_model_trained_through": str(
            permitted_through_2022["datetime"].max()
        ),
        "production_model_trained_through": str(cleaned["datetime"].max()),
        "rolling_production_model_trained_through": str(cleaned["datetime"].max()),
        "production_profile_history_end": str(cleaned["datetime"].max()),
        "trained_through": str(cleaned["datetime"].max()),
        "evaluation_training_rows": int((~train_validation["is_imputed"]).sum()),
        "production_training_rows": int((~causal_features["is_imputed"]).sum()),
        "test_period_end": str(test_history["datetime"].max()),
        "locked_test_start": str(test_history["datetime"].min()),
    }


def dataset_fingerprint(path: Path) -> str:
    """Create a quick cache key from the file metadata and program version."""

    stat = path.stat()
    message = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{PROGRAM_VERSION}"
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def load_or_train_model(
    cleaned: pd.DataFrame,
    data_path: Path,
    model_path: Path,
    n_estimators: int,
    force_retrain: bool,
) -> tuple[dict[str, Any], bool]:
    """Load a matching joblib bundle, or train and save a new one."""

    fingerprint = dataset_fingerprint(data_path)
    if model_path.exists() and not force_retrain:
        try:
            bundle = joblib.load(model_path)
            if (
                bundle.get("program_version") == PROGRAM_VERSION
                and bundle.get("dataset_fingerprint") == fingerprint
            ):
                print(f"\nLoaded cached Random Forest: {model_path}")
                return bundle, True
            print("\nThe data or program changed; retraining the cached model.")
        except Exception as exc:  # a broken cache should not block the program
            print(f"\nCould not load the cached model ({exc}); retraining.")

    trained = train_and_evaluate(cleaned, n_estimators)
    bundle = {
        **trained,
        "program_version": PROGRAM_VERSION,
        "dataset_fingerprint": fingerprint,
        "interval_minutes": INTERVAL_MINUTES,
        "target_unit": "kW",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    print(f"Saved trained Random Forest: {model_path}")
    return bundle, False


def print_evaluation(bundle: dict[str, Any]) -> None:
    evaluation = pd.DataFrame(bundle["evaluation"])
    columns = ["Split", "Method", "MAE_kW", "RMSE_kW", "MAPE_pct", "R2", "Rows"]
    print("\nMODEL EVALUATION")
    print("=" * 80)
    print(bundle["split_description"])
    tuning = pd.DataFrame(bundle.get("validation_tuning", []))
    if not tuning.empty:
        tuning_columns = [
            "Candidate",
            "Trees",
            "Max_depth",
            "Min_leaf",
            "Max_features",
            "Half_life_days",
            "MAE_kW",
            "RMSE_kW",
            "Selected",
        ]
        print("\n2022 direct-RF search (selection metric: RMSE)")
        print(
            tuning[tuning_columns].to_string(
                index=False,
                formatters={
                    "MAE_kW": "{:.4f}".format,
                    "RMSE_kW": "{:.4f}".format,
                },
            )
        )
        selected = bundle["selected_hyperparameters"]
        print(
            "Selected settings: "
            f"trees={selected['n_estimators']}, depth={selected['max_depth']}, "
            f"minimum leaf={selected['min_samples_leaf']}, "
            f"features/tree={selected['max_features']}, "
            f"recency half-life={selected['recency_half_life_days']} days"
        )

    rolling_tuning = pd.DataFrame(bundle.get("rolling_validation_tuning", []))
    if not rolling_tuning.empty:
        print("\n2022 rolling-lag RF search (selection metric: RMSE)")
        print(
            rolling_tuning[tuning_columns].to_string(
                index=False,
                formatters={
                    "MAE_kW": "{:.4f}".format,
                    "RMSE_kW": "{:.4f}".format,
                },
            )
        )
        selected_rolling = bundle["selected_rolling_hyperparameters"]
        print(
            "Selected rolling settings: "
            f"trees={selected_rolling['n_estimators']}, "
            f"depth={selected_rolling['max_depth']}, "
            f"minimum leaf={selected_rolling['min_samples_leaf']}, "
            f"features/tree={selected_rolling['max_features']}, "
            f"recency half-life={selected_rolling['recency_half_life_days']} days"
        )

    print("\nLocked validation/test metrics")
    print(
        evaluation[columns].to_string(
            index=False,
            formatters={
                "MAE_kW": "{:.4f}".format,
                "RMSE_kW": "{:.4f}".format,
                "MAPE_pct": "{:.2f}".format,
                "R2": "{:.4f}".format,
            },
        )
    )
    skipped = int(evaluation["MAPE_zero_rows_skipped"].sum())
    if skipped:
        print(f"MAPE omitted {skipped} zero-load target row(s) to avoid division by zero.")
    print("\nTWO-STAGE MODEL STATUS")
    print(
        "  Locked evaluation model: trained through "
        f"{bundle['evaluation_model_trained_through']}; 2023 remained unseen."
    )
    print(
        "  Production direct RF: refitted after evaluation on all data through "
        f"{bundle['production_model_trained_through']}."
    )
    print(
        "  Production rolling RF: refitted after evaluation on all data through "
        f"{bundle['rolling_production_model_trained_through']}."
    )
    print("  Reported 2023 metrics always come from the locked evaluation model.")
    print("\nFEATURE AVAILABILITY FOR AN ARBITRARY FUTURE DATE")
    print("  Used: calendar fields, interval number, weekend/season, and historical")
    print("        averages for the same time, weekday, day type, season, and similar days.")
    print("        Production profiles use all available history through June 2023.")
    print("  Not used: real previous 15-minute/day/week loads or rolling loads, because")
    print("            those values are unknown for a distant date such as 01/07/2024.")
    print("  Raw lag/rolling columns are still created for rolling short-horizon studies.")


def parse_prediction_date(value: str) -> pd.Timestamp:
    """Accept the requested DD/MM/YYYY format plus ISO YYYY-MM-DD."""

    value = value.strip()
    for date_format in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return pd.Timestamp(datetime.strptime(value, date_format).date())
        except ValueError:
            pass
    raise ValueError("Use DD/MM/YYYY (for example 01/07/2024) or YYYY-MM-DD.")


def normalise_method(value: str) -> str:
    simplified = re.sub(r"[^a-z]", "", value.lower())
    if simplified in {"pdf", "pdfmethod", "kde"}:
        return "pdf"
    if simplified in {"rf", "randomforest", "randomforestregression"}:
        return "rf"
    if simplified in {"both", "all", "pdfandrandomforest"}:
        return "both"
    raise ValueError("Select PDF, Random Forest, or Both.")


def forecast_selected_date(
    cleaned: pd.DataFrame,
    bundle: dict[str, Any],
    prediction_date: pd.Timestamp,
    method: str,
    historical_model_scope: str = "production",
) -> pd.DataFrame:
    """Generate exactly 96 interval forecasts for the selected date."""

    result = pd.DataFrame(
        {
            "datetime": pd.date_range(
                prediction_date.normalize(),
                periods=INTERVALS_PER_DAY,
                freq=f"{INTERVAL_MINUTES}min",
            )
        }
    )

    history_end = pd.Timestamp(bundle["production_profile_history_end"])
    permitted_history = cleaned.loc[cleaned["datetime"] <= history_end].copy()
    if permitted_history.empty:
        raise ValueError("The saved model's permitted profile history is empty.")

    locked_start = pd.Timestamp(bundle.get("locked_test_start", "2100-01-01"))
    locked_end = pd.Timestamp(bundle.get("test_period_end", "1900-01-01"))
    locked_historical_date = (
        locked_start.normalize() <= prediction_date.normalize() <= locked_end.normalize()
    )
    locked_comparison = locked_historical_date and historical_model_scope == "locked"
    comparison_history = (
        cleaned.loc[cleaned["datetime"] < locked_start].copy()
        if locked_comparison
        else permitted_history
    )

    if method in {"pdf", "both"}:
        estimator = PDFEstimator().fit(comparison_history)
        pdf_result = estimator.predict_day(prediction_date)
        result = result.merge(pdf_result, on="datetime", how="left")

    if method in {"rf", "both"}:
        future = build_future_features(comparison_history, prediction_date)
        if locked_comparison and "direct_evaluation_model" in bundle:
            direct_model = bundle["direct_evaluation_model"]
            fill_values = pd.Series(
                bundle["direct_evaluation_fill_values"], dtype=float
            )
            direct_model_type = "locked_through_2022"
        else:
            direct_model = bundle["model"]
            fill_values = pd.Series(bundle["fill_values"], dtype=float)
            direct_model_type = "production_all_history"
        matrix, _ = _prepare_matrix(
            future, bundle["feature_columns"], fill_values
        )
        prediction = np.maximum(0.0, direct_model.predict(matrix))
        result["random_forest_kw"] = prediction
        result["direct_rf_model_type"] = direct_model_type

    historical_day = cleaned.loc[
        cleaned["datetime"].dt.date == prediction_date.date(),
        ["datetime", "load_kw", "is_imputed"],
    ].copy()
    if not historical_day.empty:
        historical_day = historical_day.rename(
            columns={"load_kw": "actual_kw", "is_imputed": "actual_is_imputed"}
        )
        result = result.merge(historical_day, on="datetime", how="left")

    rolling_available = method in {"rf", "both"} and not historical_day.empty
    if rolling_available:
        causal_features = create_features(cleaned)
        rolling_day = causal_features.loc[
            causal_features["datetime"].dt.date == prediction_date.date()
        ].copy()
        if locked_comparison and "rolling_evaluation_model" in bundle:
            rolling_model = bundle["rolling_evaluation_model"]
            rolling_fill = pd.Series(bundle["rolling_fill_values"], dtype=float)
            rolling_model_type = "locked_through_2022"
        elif "rolling_production_model" in bundle:
            rolling_model = bundle["rolling_production_model"]
            rolling_fill = pd.Series(
                bundle["rolling_production_fill_values"], dtype=float
            )
            rolling_model_type = "production_all_history"
        else:
            rolling_available = False
    if rolling_available:
        rolling_x, _ = _prepare_matrix(
            rolling_day, bundle["rolling_feature_columns"], rolling_fill
        )
        rolling_prediction = np.maximum(
            0.0, rolling_model.predict(rolling_x)
        )
        rolling_result = pd.DataFrame(
            {
                "datetime": rolling_day["datetime"].to_numpy(),
                "rolling_random_forest_kw": rolling_prediction,
                "rolling_rf_model_type": rolling_model_type,
            }
        )
        result = result.merge(rolling_result, on="datetime", how="left")

    result["time"] = result["datetime"].dt.strftime("%H:%M")
    if "pdf_expected_kw" in result:
        result["pdf_energy_kwh"] = result["pdf_expected_kw"] * INTERVAL_HOURS
    if "random_forest_kw" in result:
        result["random_forest_energy_kwh"] = (
            result["random_forest_kw"] * INTERVAL_HOURS
        )
    if "rolling_random_forest_kw" in result:
        result["rolling_random_forest_energy_kwh"] = (
            result["rolling_random_forest_kw"] * INTERVAL_HOURS
        )
    if "actual_kw" in result:
        result["actual_energy_kwh"] = result["actual_kw"] * INTERVAL_HOURS
        for prediction_column, error_column in (
            ("random_forest_kw", "direct_rf_error_kw"),
            ("rolling_random_forest_kw", "rolling_rf_error_kw"),
            ("pdf_expected_kw", "pdf_error_kw"),
        ):
            if prediction_column in result:
                result[error_column] = result[prediction_column] - result["actual_kw"]
    if method == "both":
        result["rf_minus_pdf_kw"] = (
            result["random_forest_kw"] - result["pdf_expected_kw"]
        )
    return result


def _daily_summary(values: pd.Series, timestamps: pd.Series) -> dict[str, Any]:
    maximum_index = values.idxmax()
    minimum_index = values.idxmin()
    return {
        "total_energy_kwh": float(values.sum() * INTERVAL_HOURS),
        "average_load_kw": float(values.mean()),
        "maximum_load_kw": float(values.max()),
        "minimum_load_kw": float(values.min()),
        "peak_time": pd.Timestamp(timestamps.loc[maximum_index]).strftime("%I:%M %p"),
        "minimum_time": pd.Timestamp(timestamps.loc[minimum_index]).strftime("%I:%M %p"),
    }


def print_forecast(result: pd.DataFrame, prediction_date: pd.Timestamp, method: str) -> None:
    """Print daily summaries and the full 96-row forecast table."""

    print("\nFORECAST")
    print("=" * 80)
    print(f"Prediction date: {prediction_date.strftime('%A, %d %B %Y')}")
    print("Interval:        15 minutes (96 predictions)")

    summaries: dict[str, dict[str, Any]] = {}
    direct_label = "RF direct (production future-date model)"
    if (
        "direct_rf_model_type" in result
        and result["direct_rf_model_type"].iloc[0] == "locked_through_2022"
    ):
        direct_label = "RF direct (locked, trained through 2022)"
    rolling_label = "RF rolling (production all-data model)"
    if (
        "rolling_rf_model_type" in result
        and result["rolling_rf_model_type"].iloc[0] == "locked_through_2022"
    ):
        rolling_label = "RF rolling (locked, trained through 2022)"
    if "actual_kw" in result:
        summaries["Actual measured load"] = _daily_summary(
            result["actual_kw"], result["datetime"]
        )
    if method in {"pdf", "both"}:
        summaries["PDF/KDE"] = _daily_summary(
            result["pdf_expected_kw"], result["datetime"]
        )
    if method in {"rf", "both"}:
        summaries[direct_label] = _daily_summary(
            result["random_forest_kw"], result["datetime"]
        )
    if "rolling_random_forest_kw" in result:
        summaries[rolling_label] = _daily_summary(
            result["rolling_random_forest_kw"], result["datetime"]
        )

    for label, summary in summaries.items():
        print(f"\n{label}")
        print(f"  Total energy:          {summary['total_energy_kwh']:.3f} kWh")
        print(f"  Average load:          {summary['average_load_kw']:.3f} kW")
        print(f"  Maximum load:          {summary['maximum_load_kw']:.3f} kW")
        print(f"  Minimum load:          {summary['minimum_load_kw']:.3f} kW")
        print(f"  Peak time:             {summary['peak_time']}")

    if method == "both":
        pdf_total = summaries["PDF/KDE"]["total_energy_kwh"]
        rf_total = summaries[direct_label]["total_energy_kwh"]
        difference = rf_total - pdf_total
        percentage = difference / pdf_total * 100 if pdf_total else math.nan
        mean_interval_difference = float(result["rf_minus_pdf_kw"].abs().mean())
        print("\nMethod comparison")
        print(f"  RF - PDF daily energy: {difference:+.3f} kWh ({percentage:+.2f}%)")
        print(f"  Mean interval |difference|: {mean_interval_difference:.3f} kW")

    if "actual_kw" in result:
        print("\nHISTORICAL ERROR REPORT")
        actual_values = result["actual_kw"].to_numpy(dtype=float)
        actual_total = float(np.nansum(actual_values) * INTERVAL_HOURS)
        for label, column in (
            (direct_label, "random_forest_kw"),
            (rolling_label, "rolling_random_forest_kw"),
            ("PDF/KDE", "pdf_expected_kw"),
        ):
            if column not in result:
                continue
            prediction_values = result[column].to_numpy(dtype=float)
            valid = np.isfinite(actual_values) & np.isfinite(prediction_values)
            metrics = calculate_metrics(actual_values[valid], prediction_values[valid])
            predicted_total = float(np.nansum(prediction_values) * INTERVAL_HOURS)
            daily_error = predicted_total - actual_total
            daily_error_pct = daily_error / actual_total * 100 if actual_total else math.nan
            print(
                f"  {label}: MAE={metrics['MAE_kW']:.4f} kW, "
                f"MAPE={metrics['MAPE_pct']:.2f}%, daily error="
                f"{daily_error:+.3f} kWh ({daily_error_pct:+.2f}%)"
            )

    display = pd.DataFrame({"Time": result["time"]})
    if "actual_kw" in result:
        display["Actual kW"] = result["actual_kw"]
    if "pdf_expected_kw" in result:
        display["PDF kW"] = result["pdf_expected_kw"]
        display["PDF median"] = result["pdf_median_kw"]
        display["PDF mode"] = result["pdf_mode_kw"]
        display["Lower 90%"] = result["pdf_lower_kw"]
        display["Upper 90%"] = result["pdf_upper_kw"]
    if "random_forest_kw" in result:
        display["RF direct"] = result["random_forest_kw"]
    if "rolling_random_forest_kw" in result:
        display["RF rolling"] = result["rolling_random_forest_kw"]

    print("\n15-MINUTE PREDICTIONS (kW)")
    print("-" * 80)
    print(display.to_string(index=False, float_format=lambda number: f"{number:.4f}"))


def save_forecast(result: pd.DataFrame, output_dir: Path, prediction_date: pd.Timestamp, method: str) -> Path:
    """Save the detailed interval results to CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    scope_suffix = ""
    if "actual_kw" in result and "direct_rf_model_type" in result:
        model_type = str(result["direct_rf_model_type"].iloc[0])
        scope = "locked" if model_type == "locked_through_2022" else "production"
        scope_suffix = f"_{scope}"
    path = output_dir / f"forecast_{prediction_date:%Y-%m-%d}_{method}{scope_suffix}.csv"
    preferred_order = [
        "datetime",
        "time",
        "actual_kw",
        "actual_energy_kwh",
        "actual_is_imputed",
        "direct_rf_model_type",
        "rolling_rf_model_type",
        "pdf_expected_kw",
        "random_forest_kw",
        "rolling_random_forest_kw",
        "pdf_median_kw",
        "pdf_mode_kw",
        "pdf_lower_kw",
        "pdf_upper_kw",
        "pdf_sample_count",
        "pdf_similarity_basis",
        "pdf_energy_kwh",
        "random_forest_energy_kwh",
        "rolling_random_forest_energy_kwh",
        "direct_rf_error_kw",
        "rolling_rf_error_kw",
        "pdf_error_kw",
        "rf_minus_pdf_kw",
    ]
    columns = [column for column in preferred_order if column in result.columns]
    result.to_csv(path, columns=columns, index=False, float_format="%.6f")
    return path


def save_plot(result: pd.DataFrame, path: Path, prediction_date: pd.Timestamp) -> None:
    """Optionally save a simple comparison chart without creating a GUI."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(12, 5))
    if "pdf_expected_kw" in result:
        axis.plot(result["datetime"], result["pdf_expected_kw"], label="PDF expected")
        axis.fill_between(
            result["datetime"],
            result["pdf_lower_kw"],
            result["pdf_upper_kw"],
            alpha=0.18,
            label="PDF 90% interval",
        )
    if "random_forest_kw" in result:
        direct_plot_label = "RF direct (production)"
        if (
            "direct_rf_model_type" in result
            and result["direct_rf_model_type"].iloc[0] == "locked_through_2022"
        ):
            direct_plot_label = "RF direct (locked)"
        axis.plot(
            result["datetime"],
            result["random_forest_kw"],
            label=direct_plot_label,
        )
    if "rolling_random_forest_kw" in result:
        rolling_plot_label = "RF rolling (production)"
        if (
            "rolling_rf_model_type" in result
            and result["rolling_rf_model_type"].iloc[0] == "locked_through_2022"
        ):
            rolling_plot_label = "RF rolling (locked)"
        axis.plot(
            result["datetime"],
            result["rolling_random_forest_kw"],
            label=rolling_plot_label,
            linewidth=2,
        )
    if "actual_kw" in result:
        axis.plot(
            result["datetime"],
            result["actual_kw"],
            label="Actual",
            color="#111111",
            linewidth=1.5,
        )
    axis.set_title(f"UMAR load forecast - {prediction_date:%d %B %Y}")
    axis.set_ylabel("Average load (kW)")
    axis.set_xlabel("Time")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def find_default_dataset(directory: Path) -> Path:
    candidates: list[Path] = []
    for pattern in ("*.xlsx", "*.xlsm", "*.xls", "*.csv", "*.tsv"):
        candidates.extend(directory.glob(pattern))
    candidates = [path for path in candidates if not path.name.startswith("forecast_")]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("No CSV or Excel dataset was found. Use --data PATH.")
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"More than one possible dataset was found ({names}). Use --data PATH.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast 96 residential electricity-load intervals for a selected day."
    )
    parser.add_argument("--data", type=Path, help="CSV or Excel dataset path")
    parser.add_argument("--sheet", help="Excel worksheet name (auto-detected by default)")
    parser.add_argument("--date", dest="prediction_date", help="DD/MM/YYYY or YYYY-MM-DD")
    parser.add_argument("--method", help="PDF, Random Forest, or Both")
    parser.add_argument(
        "--historical-model",
        choices=("production", "locked"),
        default="production",
        help=(
            "for dates inside the dataset: use all-data production models "
            "or honest through-2022 locked models (default: production)"
        ),
    )
    parser.add_argument(
        "--unit",
        choices=("auto", "kw", "kwh"),
        default="auto",
        help="Override an ambiguous input unit (default: auto)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/random_forest_model.joblib"),
        help="joblib cache path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions"),
        help="directory for forecast CSV files",
    )
    parser.add_argument("--retrain", action="store_true", help="ignore the model cache")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=160,
        help="largest tree count considered during 2022 tuning (default: 160)",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="inspect and clean the dataset, then stop before training",
    )
    parser.add_argument("--plot", action="store_true", help="also save a PNG forecast plot")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        data_path = args.data or find_default_dataset(Path.cwd())
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        print(f"Loading {data_path} ...")
        raw, source_sheet = load_dataset(data_path, args.sheet)
        cleaned, report, mapping = clean_dataset(
            raw, data_path, source_sheet, args.unit
        )
        print_dataset_report(report, mapping)
        if args.inspect_only:
            return 0

        bundle, _ = load_or_train_model(
            cleaned=cleaned,
            data_path=data_path,
            model_path=args.model_path,
            n_estimators=args.n_estimators,
            force_retrain=args.retrain,
        )
        print_evaluation(bundle)

        date_text = args.prediction_date or input(
            "\nEnter prediction date (DD/MM/YYYY): "
        )
        prediction_date = parse_prediction_date(date_text)
        method_text = args.method or input(
            "Select method (PDF, Random Forest, or Both): "
        )
        method = normalise_method(method_text)

        historical_end = cleaned["datetime"].max().normalize()
        if prediction_date <= historical_end:
            locked_start = pd.Timestamp(bundle.get("locked_test_start", "2100-01-01"))
            locked_end = pd.Timestamp(bundle.get("test_period_end", "1900-01-01"))
            if locked_start.normalize() <= prediction_date <= locked_end.normalize():
                if args.historical_model == "locked":
                    print(
                        "Locked historical comparison: actual values will be shown. "
                        "Both RF models were trained only through 2022."
                    )
                else:
                    print(
                        "Production historical comparison: actual values will be shown, "
                        "but both RF models trained on this period. This is an in-sample "
                        "fit check, not an honest future-accuracy test."
                    )
            else:
                print(
                    "Warning: actual values will be shown, but the direct production RF "
                    "may already have trained on this historical period."
                )

        result = forecast_selected_date(
            cleaned,
            bundle,
            prediction_date,
            method,
            historical_model_scope=args.historical_model,
        )
        if len(result) != INTERVALS_PER_DAY:
            raise RuntimeError(f"Expected 96 predictions, received {len(result)}.")
        print_forecast(result, prediction_date, method)
        csv_path = save_forecast(result, args.output_dir, prediction_date, method)
        print(f"\nSaved CSV: {csv_path.resolve()}")

        if args.plot:
            plot_path = csv_path.with_suffix(".png")
            save_plot(result, plot_path, prediction_date)
            print(f"Saved plot: {plot_path.resolve()}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
