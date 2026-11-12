"""
Experiment 5: Set up a DevOps environment for ML deployment
-----------------------------------------------------------
Aim:
    * Set up a version-controlled project repository using Git.
    * Learn about the integration of ML workflows into the DevOps pipeline.

What this script does:
    1. Copies the template project (ml-devops-project/) into outputs/ml-devops-repo
       so that a fresh Git repository can be created without touching this folder.
    2. Initialises Git, makes the first commit and follows a branching workflow
       (main -> develop -> feature/*) like a real team.
    3. Runs a local DevOps pipeline: Code -> Build -> Test -> Train -> Validate -> Package -> Release
       which is exactly what the GitHub Actions file in .github/workflows/ automates.
    4. Tags the released model version (v1.0.0) and prints the Git history.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE = BASE_DIR / "ml-devops-project"
REPO = BASE_DIR / "outputs" / "ml-devops-repo"


def run(cmd, check=True):
    """Run a shell command inside the repository and print it like a terminal."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if output:
        print("   " + output.replace("\n", "\n   "))
    if check and result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}")
    return result


def git(*args, check=True):
    return run(["git", *args], check=check)


def stage(name):
    print("\n" + "-" * 70 + f"\n[PIPELINE STAGE] {name}\n" + "-" * 70)


# ---------------------------------------------------------------------------
# 1. Create a fresh copy of the project
# ---------------------------------------------------------------------------
if REPO.exists():
    shutil.rmtree(REPO, onerror=lambda f, p, e: (Path(p).chmod(0o777), f(p)))
shutil.copytree(TEMPLATE, REPO)
print("Project copied to:", REPO)

# ---------------------------------------------------------------------------
# 2. Initialise Git repository
# ---------------------------------------------------------------------------
stage("CODE - initialise version control")
git("init", "-b", "main")
# Use the student's global identity if set, otherwise a local placeholder identity
if not git("config", "user.name", check=False).stdout.strip():
    git("config", "user.name", "MLOps Student")
    git("config", "user.email", "student@example.com")
git("add", ".")
git("commit", "-m", "Initial commit: ML project structure, training code, tests and CI workflow")

# develop branch for integration, feature branch for new work
git("checkout", "-b", "develop")
git("checkout", "-b", "feature/tune-model")

# A change on the feature branch: tune hyper-parameters
train_file = REPO / "src" / "train.py"
train_file.write_text(
    train_file.read_text().replace("def train(n_estimators=100, max_depth=None)",
                                   "def train(n_estimators=200, max_depth=5)")
)
git("add", "src/train.py")
git("commit", "-m", "feat: tune RandomForest (n_estimators=200, max_depth=5)")

# ---------------------------------------------------------------------------
# 3. Run the ML pipeline on the feature branch (what CI does on every push)
# ---------------------------------------------------------------------------
stage("BUILD - check the code compiles")
run([sys.executable, "-m", "compileall", "-q", "src", "tests"])

stage("TEST - run automated unit tests")
run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])

stage("TRAIN - train the model")
run([sys.executable, "src/train.py"])

stage("VALIDATE - model quality gate")
metrics = json.loads((REPO / "models" / "metrics.json").read_text())
THRESHOLD = 0.90
print(f"accuracy={metrics['accuracy']} threshold={THRESHOLD}")
if metrics["accuracy"] < THRESHOLD:
    sys.exit("Quality gate FAILED - model will not be released")
print("Quality gate PASSED")

stage("PACKAGE - smoke test the packaged model")
run([sys.executable, "src/predict.py", "6.7", "3.0", "5.2", "2.3"])

# Note: models/ is in .gitignore, so the binary is NOT committed to Git
git("status", "--short")

# ---------------------------------------------------------------------------
# 4. Merge (pull request) and release
# ---------------------------------------------------------------------------
stage("RELEASE - merge feature -> develop -> main and tag version")
git("checkout", "develop")
git("merge", "--no-ff", "feature/tune-model", "-m", "Merge feature/tune-model into develop")
git("checkout", "main")
git("merge", "--no-ff", "develop", "-m", "Release v1.0.0")
git("tag", "-a", "v1.0.0", "-m", f"Model release v1.0.0 accuracy={metrics['accuracy']}")
git("branch", "-d", "feature/tune-model")

stage("HISTORY")
git("log", "--oneline", "--graph", "--all", "--decorate")
git("branch", "-a")
git("tag", "-n")

print("\nRepository ready at:", REPO)
print("To push to GitHub:  git remote add origin <your-repo-url>  &&  git push -u origin main --tags")
print("\nExperiment 5 completed successfully.")
