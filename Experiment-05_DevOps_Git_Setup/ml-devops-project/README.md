# ML DevOps Project (Iris Classifier)

A small, version-controlled ML project used in Experiment 5.

```
ml-devops-project/
├── src/train.py          # build step: trains model, writes models/model.joblib + metrics.json
├── src/predict.py        # serve step: loads model and predicts
├── tests/test_model.py   # quality gates run in CI
├── .github/workflows/    # GitHub Actions CI pipeline
├── requirements.txt
└── .gitignore            # keeps models/data out of Git
```

Run: `pip install -r requirements.txt && pytest -v && python src/train.py && python src/predict.py`
