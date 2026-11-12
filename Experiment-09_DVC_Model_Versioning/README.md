# Experiment 9 – Model Versioning with DVC

**Unit 3 · CO4/CO5** | Syllabus: *Install and configure DVC in a project; version control data, models and experiments for reproducibility.*

## Aim
To version datasets and models with DVC alongside Git, build a reproducible pipeline, track experiments, and roll back to an earlier version.

## Theory
Git cannot handle large data or model files. **DVC** stores them in a cache and in **remote storage** (S3, GCS, Azure, Google Drive or a local folder) and commits only small `.dvc` pointer files containing MD5 hashes.
- `dvc.yaml` defines pipeline **stages** with their deps, params and outs. `dvc repro` reruns only the stages whose inputs changed.
- `dvc exp run --set-param` tracks experiments.
- `dvc metrics diff` compares versions.

## Files
```
dvc-project/
├── params.yaml        hyperparameters
├── dvc.yaml           prepare -> train -> evaluate pipeline
└── src/get_data.py, prepare.py, train.py, evaluate.py
setup_dvc.py           runs the whole DVC workflow in outputs/dvc-workspace
```

## Run
```bash
pip install dvc pyyaml
python setup_dvc.py
```

## Commands demonstrated
```bash
git init
dvc init
dvc remote add -d localstorage <path>
dvc add data/raw/data.csv
dvc dag
dvc repro
dvc metrics show
dvc push
dvc exp run -n trees-200 --set-param train.n_estimators=200
dvc exp show
dvc status
dvc metrics diff v1.0
git checkout v1.0
dvc checkout
```
The last two commands restore data v1 and model v1 exactly.

## Result
| Version | Data rows | Accuracy | F1 |
|---|---|---|---|
| v1.0 | 400 | 0.950 | 0.962 |
| v2.0 | 569 | 0.956 | 0.966 |

Experiments `trees-200` and `depth-2` appear side by side in `dvc exp show`. After `git checkout v1.0` + `dvc checkout`, the 400-row dataset is restored.
