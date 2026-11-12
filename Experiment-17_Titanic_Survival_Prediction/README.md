# Experiment 17 – Titanic Survival Prediction using Decision Tree

**Sample project · CO3/CO4**

## Aim
To build an end-to-end survival prediction system: EDA, a feature pipeline, a tuned decision tree, evaluation, a command-line predictor and a web app.

## Method
- `features.py` → `TitanicFeatures` transformer, which learns medians on the training data only:
  - **Title** extracted from Name
  - **FamilySize**, **IsAlone**, **IsChild**
  - one-hot Title and Embarked
- `GridSearchCV` over criterion, max_depth, min_samples_leaf and class_weight, with stratified 5-fold CV.

## Run
```bash
python train.py
python predict.py
python predict.py --pclass 1 --gender female --age 29 --fare 100
python app.py
```
`app.py` serves the web form at http://127.0.0.1:5000 and the JSON API at `POST /api/predict`.

## Result
CV accuracy 0.829. On the test set: **accuracy 0.832, F1 0.776, ROC-AUC 0.864**.

| Example passenger | Survival probability |
|---|---|
| Rose (1st class woman) | 95% |
| Jack (3rd class man) | 11% |
| 6-year-old boy, 3rd class | 61% |

Outputs: EDA plots, confusion matrix, ROC curve, tree diagram and tree rules.
