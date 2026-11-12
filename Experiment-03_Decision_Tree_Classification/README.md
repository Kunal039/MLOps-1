# Experiment 3 – Decision Tree Classifier

**Unit 1 · CO3** | Syllabus: *Create a decision tree classifier, train/test on the Titanic dataset, evaluate with accuracy, confusion matrix and precision-recall.*

## Aim
To predict Titanic passenger survival with a decision tree and evaluate it using accuracy, the confusion matrix, precision, recall, F1 and the precision-recall curve.

## Theory
A decision tree splits the data on the feature that most reduces impurity:
- **Gini** = `1 − Σp²`
- **Entropy** = `−Σp log p`

`max_depth` limits growth to prevent overfitting.

The **confusion matrix** has four cells: TP, FP, FN, TN. From it:
- **Precision** = TP/(TP+FP)
- **Recall** = TP/(TP+FN)
- **F1** is their harmonic mean.

## Feature engineering
- **Title** is extracted from Name (Mr/Mrs/Miss/Master/Rare).
- **FamilySize** and **IsAlone** are derived from SibSp and Parch.
- Missing ages are filled with the median age for each title.

## Run
```bash
python decision_tree.py
```

## Result (test set)
Accuracy ≈ **0.81**, precision 0.84, recall 0.62. The most important features are Title_Mr, Fare and Pclass. Outputs include the confusion matrix with the PR curve, a tree diagram, feature importance and the tree rules as text.
