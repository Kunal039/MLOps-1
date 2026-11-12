"""
Experiment 11: Mini Project Phase-1 - Problem definition, data collection and EDA
==================================================================================
Project : Household Electricity Consumption Prediction System
Dataset : UCI "Individual Household Electric Power Consumption"
          https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Data collection
    If the real UCI file `data/household_power_consumption.txt` exists it is used.
    Otherwise a dataset with the SAME format (minute-level readings, ';' separator,
    '?' for missing values) is generated so the whole project runs offline.
    To use the real data: download the zip from the link above and extract the .txt into data/.

Output
    data/household_power_consumption.txt (if generated)
    outputs/*.png  - EDA plots
    outputs/eda_summary.txt
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "household_power_consumption.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data collection
# ---------------------------------------------------------------------------
def generate_uci_like_data(days=180, seed=42):
    """Create minute-level household power data that follows realistic daily/weekly/seasonal patterns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2009-01-01", periods=days * 24 * 60, freq="min")
    hour = idx.hour + idx.minute / 60
    doy = idx.dayofyear.to_numpy()

    # Daily profile: low at night, peak in the morning (7-9) and a bigger peak in the evening (18-22)
    daily = (0.35
             + 0.9 * np.exp(-((hour - 8) ** 2) / 2.5)
             + 1.6 * np.exp(-((hour - 20) ** 2) / 4.0))
    weekend = np.where(idx.dayofweek >= 5, 1.25, 1.0)                  # more use at home on weekends
    seasonal = 1 + 0.35 * np.cos(2 * np.pi * (doy - 15) / 365)          # heating in winter
    appliance_spikes = rng.exponential(0.25, len(idx)) * (rng.random(len(idx)) < 0.08) * 8

    active = np.clip(daily * weekend * seasonal + appliance_spikes + rng.normal(0, 0.12, len(idx)), 0.08, None)
    reactive = np.clip(0.1 + 0.05 * active + rng.normal(0, 0.03, len(idx)), 0, None)
    voltage = 240.8 - 1.8 * active + rng.normal(0, 1.2, len(idx))
    intensity = active * 1000 / voltage
    kitchen = np.where((rng.random(len(idx)) < 0.05) & ((hour > 11) & (hour < 21)), rng.uniform(1, 38, len(idx)), 0)
    laundry = np.where(rng.random(len(idx)) < 0.04, rng.uniform(1, 30, len(idx)), rng.choice([0, 1, 2], len(idx)))
    heater = np.clip(active * 1000 / 60 * 0.45 + rng.normal(0, 1, len(idx)), 0, 31)

    df = pd.DataFrame({
        "Date": idx.strftime("%d/%m/%Y").str.replace(r"^0", "", regex=True).str.replace(r"/0", "/", regex=True),
        "Time": idx.strftime("%H:%M:%S"),
        "Global_active_power": active.round(3),
        "Global_reactive_power": reactive.round(3),
        "Voltage": voltage.round(2),
        "Global_intensity": intensity.round(1),
        "Sub_metering_1": kitchen.round(0),
        "Sub_metering_2": laundry.round(0),
        "Sub_metering_3": heater.round(0),
    })

    # Real-world problems that preprocessing (Phase 2) must handle
    value_cols = df.columns[2:]
    df = df.astype({c: object for c in value_cols})
    gap_starts = rng.choice(len(df) - 600, 25, replace=False)            # meter outages
    for s in gap_starts:
        df.iloc[s:s + rng.integers(10, 500), 2:] = "?"
    outliers = rng.choice(len(df), 40, replace=False)                    # faulty readings
    df.loc[outliers, "Global_active_power"] = rng.uniform(15, 30, 40).round(3)
    return df


if DATA_FILE.exists():
    print("Using existing dataset:", DATA_FILE.name)
else:
    print("Real UCI file not found - generating a UCI-format dataset (180 days, minute-level)...")
    DATA_FILE.parent.mkdir(exist_ok=True)
    generate_uci_like_data().to_csv(DATA_FILE, sep=";", index=False)
    print("Saved:", DATA_FILE)

# ---------------------------------------------------------------------------
# 2. Load and understand the data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE, sep=";", na_values=["?"], low_memory=False)
df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
df = df.drop(columns=["Date", "Time"]).set_index("Datetime").sort_index()

summary = []


def report(text=""):
    print(text)
    summary.append(str(text))


report("=" * 70)
report("PHASE 1 - EXPLORATORY DATA ANALYSIS")
report("=" * 70)
report(f"Rows (minute readings): {len(df):,}")
report(f"Period: {df.index.min()}  ->  {df.index.max()}  ({(df.index.max() - df.index.min()).days} days)")
report(f"Columns: {list(df.columns)}")
report("\nStatistical summary:")
report(df.describe().T.round(3).to_string())

missing = df.isna().sum()
report("\nMissing values per column:")
report(missing.to_string())
report(f"Missing rows: {df.isna().any(axis=1).sum():,} ({df.isna().any(axis=1).mean():.2%})")

q1, q3 = df["Global_active_power"].quantile([0.25, 0.75])
upper = q3 + 3 * (q3 - q1)
report(f"\nPossible outliers in Global_active_power (> {upper:.2f} kW): {(df['Global_active_power'] > upper).sum()}")

hourly = df["Global_active_power"].resample("h").mean()
report("\nTarget for the project: hourly average Global_active_power (kW)")
report(f"Hourly mean = {hourly.mean():.3f} kW, max = {hourly.max():.3f} kW")

# ---------------------------------------------------------------------------
# 3. EDA plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
df["Global_active_power"].resample("D").mean().plot(ax=axes[0, 0], title="Daily average active power (kW)")
df.groupby(df.index.hour)["Global_active_power"].mean().plot(
    kind="bar", ax=axes[0, 1], title="Average power by hour of day", color="tab:orange")
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_day = df.groupby(df.index.dayofweek)["Global_active_power"].mean()
axes[1, 0].bar(days, by_day.values, color="tab:green")
axes[1, 0].set_title("Average power by day of week")
df["Global_active_power"].clip(upper=8).plot(kind="hist", bins=60, ax=axes[1, 1],
                                             title="Distribution of active power")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_patterns.png")
plt.close()

plt.figure(figsize=(8, 6))
corr = df.corr()
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.colorbar()
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
plt.close()

plt.figure(figsize=(14, 4))
week = df["Global_active_power"].loc[df.index.min():df.index.min() + pd.Timedelta(days=7)]
week.resample("15min").mean().plot(title="One week of readings (15-min average) - gaps are missing data")
plt.ylabel("kW")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "one_week.png")
plt.close()

report("\nKey observations:")
report("* Strong daily cycle: low at night, morning peak and a larger evening peak.")
report("* Weekends use more power than weekdays; winter months use more than summer.")
report("* Global_intensity is almost perfectly correlated with active power (it is derived from it),")
report("  so it must NOT be used as an input feature (target leakage).")
report("* Missing blocks (meter outages) and extreme spikes must be handled in Phase 2.")
(OUTPUT_DIR / "eda_summary.txt").write_text("\n".join(summary))
print("\nPlots and summary saved to", OUTPUT_DIR)
print("Experiment 11 (Mini Project Phase-1) completed successfully.")
