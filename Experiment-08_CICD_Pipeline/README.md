# Experiment 8 – CI/CD Pipeline for ML Models

**Unit 3 · CO5** | Syllabus: *Use Jenkins or GitHub Actions to automate testing and deployment of ML models; set up continuous integration for training and validation.*

## Aim
To automate lint → test → train → validate → build → deploy, so that only models passing a quality gate are deployed.

## Theory
- **CI** runs on every push: code checks, unit tests (data, model and API tests) and automated training.
- A **model quality gate** (`validate_model.py`) exits with code 1 if accuracy or F1 is below the thresholds in `params.json`. That failure blocks deployment.
- **CD** builds a Docker image from the *validated* model artifact, smoke-tests it, pushes it to a registry and deploys it.

## Files
```
src/data.py            load + validate data (schema, missing values, classes)
src/train.py           train with params.json -> model/, reports/metrics.json
src/validate_model.py  quality gate
app.py                 Flask service that gets deployed
tests/test_pipeline.py 8 tests: data, model, API
.github/workflows/ml-cicd.yml   GitHub Actions (CI + CD jobs)
Jenkinsfile            same pipeline for Jenkins
Dockerfile, params.json, .flake8
run_pipeline.py        runs all stages locally like a CI server
```

## Run
```bash
pip install -r requirements.txt
python run_pipeline.py
```

To run on GitHub, push this folder as a repository; the workflow then runs in the **Actions** tab. GitHub only reads workflows from the root `.github/workflows/` folder.

**Try it:** set `"min_accuracy": 1.01` in `params.json` and rerun. The pipeline stops at the gate.

## Result
All 5 stages pass. CV accuracy is 0.986 and test accuracy 1.0 on the Wine dataset, and the gate passes.
