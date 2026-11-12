# Experiment 12 – Mini Project Phase 2: Preprocessing & Feature Engineering

> **Mini project:** *Household Electricity Consumption Prediction System* (your proposal: Kunal Mittal, Ansh Kumar, Vishal Kumar).
> Phases 1→5 run in order (Experiments 11 → 15). To run everything at once: `python Experiment-15_MiniProject_Phase5_CICD_Monitoring/run_full_pipeline.py`.

## Aim
To clean the raw minute data, convert it to an hourly forecasting dataset with lag features, and split it by time.

## Steps
1. Parse the datetime and complete the minute index. Convert `?` to NaN.
2. **Outliers:** readings above the IQR fence (max 11 kW, the household supply limit) become missing.
3. **Missing values:** time interpolation for gaps under 1 hour, and the same minute last week for longer gaps.
4. **Resample** to hourly means. The target is hourly Global_active_power.
5. **Features**, using past values only (no leakage):
   - hour, day_of_week, month, is_weekend
   - cyclic hour_sin and hour_cos
   - lags at 1, 2, 3, 24 and 168 h
   - rolling mean and std over the last 3 and 24 hours
   - previous-hour voltage, reactive power and sub_metering_3
6. **Time-based split** 70/15/15 into train, validation and test. Time series are never shuffled.
7. Fit `StandardScaler` on the training data only.

## Run
```bash
python phase2_preprocessing.py
```
Run Phase 1 first.

## Result
4,152 hourly rows split into 2,906 train / 623 validation / 623 test. The features most correlated with the target are lag_168 (0.99), lag_24 (0.97) and lag_1 (0.81).
