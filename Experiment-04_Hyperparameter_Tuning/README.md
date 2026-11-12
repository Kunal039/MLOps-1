# Experiment 4 – Hyperparameter Tuning with GridSearchCV

**Unit 2 · CO3** | Syllabus: *Use GridSearchCV to tune a Decision Tree classifier; compare tuned vs default model.*

## Aim
To improve a decision tree by searching its hyperparameters with cross-validation, and to compare the tuned model with the default one.

## Theory
- **Hyperparameters** (max_depth, min_samples_split, min_samples_leaf, criterion, max_features) are set before training.
- **GridSearchCV** tries every combination using **k-fold cross-validation** (here stratified 5-fold) and keeps the best one.
- A default tree grows until it memorizes the training data (**overfitting**). Limiting depth fixes this.
- The **validation curve** (train vs CV accuracy against max_depth) shows the transition from underfitting to overfitting.

## Run
```bash
python hyperparameter_tuning.py
```
This searches 448 combinations × 5 folds = 2,240 fits.

## Result
| | Default | Tuned |
|---|---|---|
| Train accuracy | 0.985 | 0.840 |
| Test accuracy | 0.804 | **0.832** |
| F1 | 0.733 | **0.776** |
| ROC-AUC | 0.775 | **0.864** |

The train-test gap falls from 0.18 to 0.007, so tuning removed the overfitting. Best parameters: `gini, max_depth=3, min_samples_split=20`.
