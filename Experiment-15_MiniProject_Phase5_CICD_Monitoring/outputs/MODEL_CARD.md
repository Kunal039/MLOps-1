# Model Card - Household Electricity Consumption Predictor

| Item | Value |
|---|---|
| Model version | v1 |
| Algorithm | Random Forest {'max_depth': 12, 'min_samples_leaf': 5, 'n_estimators': 100} |
| Task | Predict next-hour household active power (kW) |
| Training data | UCI Household Electric Power Consumption (hourly aggregated), 3529 hours |
| Test RMSE / MAE / R2 | 0.0854 / 0.0673 / 0.94 |
| Baseline RMSE (same hour yesterday) | 0.1202 |

## Intended use
Helping a household plan electricity use and helping utilities balance load.
**Not** for billing, credit decisions, insurance, or inferring the behaviour of residents.

## Inputs
Last 168 hours of consumption, calendar features and last-hour voltage / reactive power / sub-meter.

## Performance across groups
```
                 group    MAE  hours  ratio_to_overall
     Day type: Weekday 0.0666    431              0.99
     Day type: Weekend 0.0687    192              1.02
    Time of day: Night 0.0597    155              0.89
  Time of day: Morning 0.0703    156              1.04
Time of day: Afternoon 0.0712    156              1.06
  Time of day: Evening 0.0678    156              1.01
```

## Limitations
* Trained on one household; does not generalise to other homes without retraining.
* Sudden behaviour changes (new appliance, holidays, guests) cause drift - monitored via live MAE.
* No weather data is used.

## Ethical considerations
* Privacy: data is pseudonymised and aggregated to hourly values; raw minute data is not stored by the API.
* Consent: households must opt in; data retention is limited and deletable on request.
* Security: inputs are validated, the container runs as non-root, model files are checksum-verified.
