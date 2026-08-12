import unittest

import numpy as np
import pandas as pd

import energy_forecast as forecast


class ForecastTests(unittest.TestCase):
    @staticmethod
    def synthetic_history(days: int = 90) -> pd.DataFrame:
        timestamps = pd.date_range("2021-01-01", periods=days * 96, freq="15min")
        slot = timestamps.hour * 4 + timestamps.minute // 15
        load = 0.3 + 0.12 * np.sin(2 * np.pi * slot / 96) + 0.03 * (
            timestamps.dayofweek >= 5
        )
        return pd.DataFrame(
            {
                "datetime": timestamps,
                "load_kw": load,
                "energy_kwh": load * 0.25,
                "valid_samples": 15,
                "is_imputed": False,
            }
        )

    def test_date_parsing(self):
        self.assertEqual(
            forecast.parse_prediction_date("01/07/2024"), pd.Timestamp("2024-07-01")
        )

    def test_features_do_not_use_future_target(self):
        history = self.synthetic_history(20)
        original = forecast.create_features(history)
        changed = history.copy()
        changed.loc[500:, "load_kw"] = changed.loc[500:, "load_kw"] * 10
        recalculated = forecast.create_features(changed)
        columns = [
            "lag_15min_kw",
            "lag_previous_day_kw",
            "lag_previous_week_kw",
            "rolling_mean_1h_kw",
            "historical_similar_mean_kw",
        ]
        pd.testing.assert_frame_equal(
            original.loc[:499, columns], recalculated.loc[:499, columns]
        )

    def test_pdf_produces_96_ordered_nonnegative_intervals(self):
        history = self.synthetic_history()
        # Add a day-to-day change so the KDE integration path is exercised,
        # rather than the zero-variance fallback for each interval.
        history["load_kw"] += history["datetime"].dt.dayofyear.to_numpy() * 0.0005
        estimator = forecast.PDFEstimator(min_samples=2).fit(history)
        result = estimator.predict_day(pd.Timestamp("2024-07-01"))
        self.assertEqual(len(result), 96)
        self.assertTrue((result["pdf_expected_kw"] >= 0).all())
        self.assertTrue((result["pdf_lower_kw"] <= result["pdf_upper_kw"]).all())

    def test_future_feature_shape(self):
        future = forecast.build_future_features(
            self.synthetic_history(), pd.Timestamp("2024-07-01")
        )
        self.assertEqual(len(future), 96)
        self.assertTrue(set(forecast.RF_FEATURE_COLUMNS).issubset(future.columns))
        self.assertFalse(future[forecast.RF_FEATURE_COLUMNS].isna().any().any())

    def test_requested_targets_cannot_change_direct_profile_features(self):
        full = self.synthetic_history(40)
        history = full.iloc[: 30 * 96].copy()
        requested = full.iloc[30 * 96 :].copy()
        original = forecast.build_profile_features(history, requested)
        changed = requested.copy()
        changed["load_kw"] = 999.0
        recalculated = forecast.build_profile_features(history, changed)
        profile_columns = [
            "historical_same_time_mean_kw",
            "historical_same_dow_mean_kw",
            "historical_similar_mean_kw",
            "historical_season_mean_kw",
            "historical_daytype_mean_kw",
        ]
        pd.testing.assert_frame_equal(
            original[profile_columns], recalculated[profile_columns]
        )

    def test_rolling_features_never_use_the_current_target(self):
        history = self.synthetic_history(20)
        original = forecast.create_features(history)
        changed = history.copy()
        changed.loc[500, "load_kw"] = 999.0
        recalculated = forecast.create_features(changed)
        rolling_columns = [
            "lag_15min_kw",
            "lag_previous_day_kw",
            "lag_previous_week_kw",
            "rolling_mean_1h_kw",
            "rolling_mean_24h_kw",
        ]
        pd.testing.assert_series_equal(
            original.loc[500, rolling_columns],
            recalculated.loc[500, rolling_columns],
        )
        # The changed target becomes visible only to the following row's lag.
        self.assertNotEqual(
            original.loc[501, "lag_15min_kw"],
            recalculated.loc[501, "lag_15min_kw"],
        )


if __name__ == "__main__":
    unittest.main()
