# Wine Quality — CS643 Project

This repository contains scripts for training and evaluating a Spark ML pipeline on the Wine Quality dataset.

**Repository layout**
- `train.py`: Training script that renames CSV columns to `f0..fN` and `quality`, tunes `LogisticRegression`, and saves a `PipelineModel`.
- `predict.py`: Loads a saved `PipelineModel`, renames test CSV columns to `f0..fN`/`quality`, runs predictions, and prints an `F1=` line.
- `code/train_model.py`: Alternate training script (RandomForest) with a different CLI and slightly different preprocessing.
- `model_lr/`: Example saved model directory (contains pipeline metadata and stages).
- `TrainingDataset.csv`, `ValidationDataset.csv`: example datasets used by the scripts.
- `requirements.txt`: Python dependencies (needs cleanup — see Notes).
- `Dockerfile`: Container image for running prediction (needs JAVA_HOME fix — see Notes).

**Prerequisites**
- Java 11+ (PySpark needs a JVM). Ensure `java -version` works in your environment.
- Python 3.8+ and `pip`.
- PySpark compatible with your Java/Python versions. This repo lists `pyspark==3.5.1` in `requirements.txt`.

**Install (recommended local/dev)**

1. Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Fix `requirements.txt` (it currently contains code fences). The expected content should look like:

```
pyspark
numpy
pandas
```

Save that exact text to `requirements.txt`, then install:

```powershell
pip install -r requirements.txt
```

**Run training**

Using the primary `train.py` (per-repo example, does hyperparam loop over LogisticRegression):

```powershell
# using spark-submit (recommended when running a real Spark cluster / local spark)
spark-submit train.py TrainingDataset.csv ValidationDataset.csv model_lr

# or with installed pyspark you may run with plain python
python train.py TrainingDataset.csv ValidationDataset.csv model_lr
```

Using the alternate script `code/train_model.py` (RandomForest):

```powershell
python code\train_model.py TrainingDataset.csv ValidationDataset.csv model_lr
# or:
# spark-submit code/train_model.py TrainingDataset.csv ValidationDataset.csv model_lr
```

Note: `train.py` renames all feature columns to `f0..fN` and the last column to `quality`. `code/train_model.py` currently does not perform that same renaming — see Known Issues.

**Run prediction / evaluation**

```powershell
spark-submit predict.py model_lr ValidationDataset.csv
# or
python predict.py model_lr ValidationDataset.csv
```

The script prints `F1=<score>` as its primary output.

**Docker**

The included `Dockerfile` builds a prediction container. Before building:

- Ensure `requirements.txt` is fixed (remove markdown/``` fences).
- The `Dockerfile` sets `JAVA_HOME` to an `arm64` path. On most `python:3.11-slim` images (amd64) Java is installed under `/usr/lib/jvm/java-21-openjdk-amd64`. If your build fails, update `JAVA_HOME` in the `Dockerfile` to match the installed JRE path.

Build and run example:

```powershell
# from repo root
docker build -t wine-predict:latest .

docker run --rm wine-predict:latest
```

**Known issues & recommended fixes**
- `requirements.txt` currently contains triple-backtick fences (looks like a Markdown snippet). That will break `pip install -r requirements.txt`. Replace it with plain package lines (see Install section).
- `Dockerfile` `JAVA_HOME` path is architecture-specific. Adjust to your platform or set it dynamically.
- Inconsistent preprocessing across scripts: `train.py` renames columns to `f0..fN` while `code/train_model.py` omits that step. This can make saved pipelines incompatible with `predict.py` if input column names differ. I recommend extracting the rename/normalize logic into a shared helper (e.g., `code/preprocess.py`) and using it from all scripts.
- Consider pinning `numpy` and `pandas` versions in `requirements.txt` for reproducibility.
- Add a short `CONTRIBUTING` or `RUNNING.md` if multiple users will work on this project.

**Next steps I can take (if you want)**
- Fix `requirements.txt` and update `Dockerfile` `JAVA_HOME` (quick, low-risk).
- Add a small `code/preprocess.py` and update `train.py`/`predict.py`/`code/train_model.py` to use it (safe refactor to enforce compatibility).
- Add a minimal test script or GitHub Actions workflow to run a smoke test.

If you want me to apply any of those changes now, tell me which and I will proceed.

---

Created by the repository review assistant. If you'd like the README adjusted (more examples, diagrams, or specific versions), tell me which parts to expand.
