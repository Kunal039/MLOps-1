# Experiment 5 – DevOps Environment for ML (Git)

**Unit 2 · CO5** | Syllabus: *Set up a version-controlled project repository using Git; learn the integration of ML workflows into the DevOps pipeline.*

## Aim
To create a Git-managed ML project and run its workflow as a DevOps pipeline: Code → Build → Test → Train → Validate → Package → Release.

## Theory
- **Git** tracks code history.
- A **branching model** separates work: `main` (production), `develop` (integration) and `feature/*` (new work).
- ML adds steps that normal DevOps lacks: data validation, training, **model quality gates** and model packaging.
- Large binaries (models, data) are kept out of Git with `.gitignore` and versioned with DVC instead (see Experiment 9).
- **CI** (GitHub Actions) re-runs tests and training on every push.

## Files
```
ml-devops-project/            template project
├── src/train.py, src/predict.py
├── tests/test_model.py       quality gates (accuracy ≥ 0.90)
├── .github/workflows/ml-ci.yml
├── requirements.txt, .gitignore, README.md
setup_devops_repo.py          creates outputs/ml-devops-repo and runs the whole Git + pipeline workflow
```

## Run
```bash
python setup_devops_repo.py
```
The script copies the template into `outputs/ml-devops-repo` and then runs, in order:
1. `git init`
2. First commit
3. Create `develop` and `feature/tune-model` branches
4. Commit a hyperparameter change
5. Run the pipeline stages
6. Merge with `--no-ff`
7. Tag `v1.0.0`
8. Print `git log --graph`

To publish the repo:
```bash
cd outputs/ml-devops-repo
git remote add origin <url>
git push -u origin main --tags
```
