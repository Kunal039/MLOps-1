# Experiment 13 – Mini Project Phase 3: Model Building, Tuning & Selection

> **Mini project:** *Household Electricity Consumption Prediction System* (your proposal: Kunal Mittal, Ansh Kumar, Vishal Kumar).
> Phases 1→5 run in order (Experiments 11 → 15). To run everything at once: `python Experiment-15_MiniProject_Phase5_CICD_Monitoring/run_full_pipeline.py`.

## Aim
To train, tune and compare the syllabus algorithms and select the best model for deployment.

## Method
- **Baseline:** "same hour yesterday" (lag_24). A model must beat it to be useful.
- **Models:** Linear Regression, Ridge, Decision Tree, Random Forest, SVR, KNN. Each runs in a `Pipeline(StandardScaler → model)`.
- **Tuning:** `GridSearchCV` with **TimeSeriesSplit** (5 folds), so each fold validates on data that comes after its training data.
- **Selection:** lowest validation RMSE. The train-vs-validation gap is checked for overfitting. The chosen model is retrained on train + validation and evaluated once on test.

## Run
```bash
python phase3_model_building.py
```

## Result (validation set)
| Model | Val RMSE | Val MAE | R² |
|---|---|---|---|
| **Random Forest** | **0.083** | 0.067 | 0.957 |
| SVR | 0.087 | 0.071 | 0.953 |
| Linear / Ridge | 0.088 | 0.069 | 0.952 |
| Decision Tree | 0.091 | 0.072 | 0.949 |
| KNN | 0.108 | 0.086 | 0.927 (overfits: train RMSE 0) |

On the test set, Random Forest scores **RMSE 0.085, MAE 0.067, R² 0.94**. That is **29% better than the baseline** (RMSE 0.120). The model and its metadata are saved to `outputs/models/`.
