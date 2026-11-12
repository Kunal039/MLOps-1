# Experiment 11 – Mini Project Phase 1: Data Collection & EDA

> **Mini project:** *Household Electricity Consumption Prediction System* (your proposal: Kunal Mittal, Ansh Kumar, Vishal Kumar).
> Phases 1→5 run in order (Experiments 11 → 15). To run everything at once: `python Experiment-15_MiniProject_Phase5_CICD_Monitoring/run_full_pipeline.py`.

## Aim
To collect the household power dataset, understand its structure and quality, and find the patterns that will guide feature engineering.

## Data
UCI *Individual Household Electric Power Consumption*: minute-level readings with columns Global_active_power, reactive power, voltage, intensity and 3 sub-meters. `?` marks a missing value.
- **Real data:** download it from https://archive.ics.uci.edu/dataset/235 and put `household_power_consumption.txt` in `data/`.
- **No file present:** the script generates a dataset in the same format (180 days, 259,200 rows) with realistic daily, weekly and seasonal patterns, meter outages and faulty spikes. This lets the project run offline.

## Run
```bash
python phase1_data_collection_eda.py
```

## Findings (`outputs/eda_summary.txt`)
- 2.3% of rows are missing, in blocks caused by meter outages. There are also extreme faulty readings.
- There is a strong daily cycle: a morning peak around 08:00 and a larger evening peak around 20:00. Weekends and winter months use more power.
- `Global_intensity` is almost exactly proportional to active power, so it is **target leakage** and must not be used as a feature.

Plots: `eda_patterns.png`, `correlation_matrix.png`, `one_week.png`.
