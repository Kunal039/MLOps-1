"""
Build the model's feature vector from a prediction request.
Must create exactly the same features as Phase-2 (phase2_preprocessing.py),
otherwise the deployed model would receive different inputs than it was trained on
(training-serving skew).
"""

from datetime import datetime

import numpy as np
import pandas as pd

HISTORY_HOURS = 168   # one week of hourly consumption is needed for the lag_168 feature


def build_features(payload, feature_order):
    """
    payload = {
        "timestamp": "2009-06-20 18:00:00",            # hour to predict
        "recent_consumption": [... 168 values ...],     # hourly kW, oldest first, last = previous hour
        "voltage_last_hour": 239.4,
        "reactive_power_last_hour": 0.15,
        "sub_metering_3_last_hour": 7.2
    }
    """
    history = payload.get("recent_consumption")
    if not isinstance(history, list) or len(history) < HISTORY_HOURS:
        raise ValueError(f"'recent_consumption' must contain at least {HISTORY_HOURS} hourly values")
    history = np.asarray(history[-HISTORY_HOURS:], dtype=float)
    if (history < 0).any():
        raise ValueError("Consumption values cannot be negative")

    ts = pd.Timestamp(payload.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:00:00"))
    last_24 = pd.Series(history[-24:])

    features = {
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "month": ts.month,
        "is_weekend": int(ts.dayofweek >= 5),
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "lag_1": history[-1],
        "lag_2": history[-2],
        "lag_3": history[-3],
        "lag_24": history[-24],
        "lag_168": history[-168],
        "rolling_mean_3": history[-3:].mean(),
        "rolling_mean_24": last_24.mean(),
        "rolling_std_24": last_24.std(),          # pandas std (ddof=1) - same as in Phase-2
        "Voltage_lag_1": float(payload.get("voltage_last_hour", 240.0)),
        "Global_reactive_power_lag_1": float(payload.get("reactive_power_last_hour", 0.12)),
        "Sub_metering_3_lag_1": float(payload.get("sub_metering_3_last_hour", 6.0)),
    }
    return pd.DataFrame([features])[feature_order]
