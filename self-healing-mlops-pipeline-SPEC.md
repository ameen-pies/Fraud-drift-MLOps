# Self-Healing Model Pipeline — Build Spec

> Automated ML GitOps & Drift Monitoring Pipeline
> Target: 3–4 weeks, fully local, zero cloud cost, CV-ready with hard metrics.

This doc is written to be fed directly to a coding agent (Claude Code, Cursor, etc.) section by section. Each week is scoped so you can hand the agent one "Week N" block at a time, review the diff, then move on. Don't dump the whole doc in one prompt — you'll get a shallower result. Feed it week by week, and after each week run the acceptance checks before moving on.

---

## 0. Project Overview

**What you're building:** an API that serves a fraud-detection model, wrapped in:
1. A CI/CD pipeline that blocks bad model deployments (accuracy gate).
2. A drift-detection layer that watches "live" traffic and flags when the model is going stale.
3. A closed loop: drift detected → retrain → register new model version → ready for rollback.

**Non-negotiable design decisions (lock these in before coding):**
- Everything runs on `docker compose` on your laptop. No AWS/GCP/Azure account needed.
- MLflow tracking server runs locally (SQLite backend + local artifact store) via Docker.
- The "live traffic" is simulated by a script that perturbs your held-out test set — you are not going to stand up a real production feed for this.
- One model family only: binary classification (fraud / not fraud). Don't scope-creep into multi-model.

**Repo structure to have the agent create first:**

```
self-healing-mlops/
├── docker-compose.yml
├── .github/workflows/ci.yml
├── api/
│   ├── main.py              # FastAPI app
│   ├── model_loader.py
│   ├── schemas.py
│   ├── Dockerfile
│   └── requirements.txt
├── training/
│   ├── train.py             # trains + logs to MLflow
│   ├── data_prep.py
│   └── requirements.txt
├── monitoring/
│   ├── simulate_traffic.py  # generates drifted requests, hits the API
│   ├── drift_check.py       # Evidently AI report + threshold logic
│   ├── retrain_trigger.py   # closes the loop
│   └── requirements.txt
├── tests/
│   ├── test_api.py
│   └── test_model_quality.py
├── data/
│   ├── raw/                 # gitignored, or a small sample committed
│   └── reference/           # baseline distribution snapshot for drift comparison
├── mlruns/                  # gitignored — MLflow local artifact store
├── .gitignore
└── README.md
```

Have the agent scaffold this whole tree with empty/stub files in one shot before writing logic — makes every subsequent prompt shorter because the agent has the map.

---

## 1. Dataset & Model (do this before Week 1 coding starts)

Use **Kaggle's Credit Card Fraud Detection dataset** (or the synthetic `sklearn.datasets.make_classification` if you want zero download friction — recommend starting with synthetic, swap to real Kaggle data once the pipeline works, so you're not debugging data issues and pipeline issues at once).

- Model: `RandomForestClassifier` from scikit-learn. Nothing fancier. The infra is the star, not the model.
- Split: train / validation / "reference" (for drift baseline) / held-out "future" set (source for simulated drifted traffic).
- Target baseline accuracy: aim for something in the 90–97% range on a fraud dataset (these are usually imbalanced — track **precision/recall/F1**, not just accuracy, and say so honestly on your CV. Using only "accuracy" on an imbalanced fraud dataset is a red flag to anyone who's done ML before — a model that always predicts "not fraud" gets 99%+ "accuracy" and is useless).

**Correction to the source material:** the original brief above uses "85% accuracy threshold" as the CI/CD gate metric. For a fraud/imbalanced dataset, gate on **F1 or recall on the minority class** instead, and say so explicitly in your README. This one change signals you actually understand the domain instead of copy-pasting a generic MLOps tutorial.

---

## 2. Week 1 — Base Model + FastAPI + Docker

### Goals
- Trained model artifact saved to disk.
- `/predict` endpoint serving it.
- Whole thing runs via `docker compose up`.

### Tasks for the agent
1. `training/data_prep.py`: load dataset, split, save `data/reference/reference_data.csv` (the baseline distribution Evidently will compare against later).
2. `training/train.py`: train `RandomForestClassifier`, evaluate (accuracy, precision, recall, F1, confusion matrix), save model to `models/model_v1.pkl` (or straight into MLflow — see Week 2, but a plain pickle is fine for Week 1).
3. `api/schemas.py`: Pydantic model for the `/predict` request body matching your feature columns exactly (name every column — don't accept arbitrary JSON).
4. `api/model_loader.py`: loads the pickled model once at startup, not per-request.
5. `api/main.py`: FastAPI app with:
   - `GET /health` → `{"status": "ok"}`
   - `POST /predict` → takes validated schema, returns `{"prediction": 0|1, "fraud_probability": float}`
   - Log every request/response pair to a local file or SQLite table (`logs/predictions.jsonl`) — you'll need this history for drift detection in Week 3.
6. `api/Dockerfile`: slim Python base image, install `requirements.txt`, copy app, `uvicorn` entrypoint.
7. `docker-compose.yml`: single `api` service for now, exposing port 8000.

### Acceptance check (run yourself, don't just trust the agent)
```bash
docker compose up --build
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...your feature payload...}'
```
You should get a real prediction back, and a line appended to `logs/predictions.jsonl`.

---

## 3. Week 2 — MLflow Tracking + CI/CD Accuracy Gate

### Goals
- Every training run is logged to MLflow (params, metrics, model artifact).
- A GitHub Action runs on every push, retrains/evaluates, and **fails the build** if quality drops below your threshold.

### Tasks for the agent
1. Add an `mlflow` service to `docker-compose.yml`:
   - Backend store: SQLite file (`mlruns.db`), mounted volume so it persists.
   - Artifact store: local `./mlartifacts` directory, mounted volume.
   - Exposed on port 5000.
2. Update `training/train.py` to:
   - `mlflow.set_tracking_uri(...)` pointing at the local server.
   - `mlflow.start_run()` wrapping the training.
   - `mlflow.log_params(...)` for model hyperparameters.
   - `mlflow.log_metrics(...)` for accuracy/precision/recall/F1.
   - `mlflow.sklearn.log_model(...)` to register the model artifact.
   - Also **register** the model in the MLflow Model Registry (not just log it) so version bumps (`v1`, `v2`) are explicit and visible — this is the detail that makes "model versioning" a true claim on your CV instead of a buzzword.
3. `tests/test_model_quality.py`:
   - Loads the latest registered model.
   - Runs it against a held-out validation set.
   - **Asserts** F1 (or recall on minority class — see Section 1 correction) is above your chosen threshold.
   - This is a real `pytest` test, not a print statement — CI needs a non-zero exit code to fail the build.
4. `.github/workflows/ci.yml`:
   - Trigger: `on: push` and `on: pull_request`.
   - Steps: checkout → set up Python → install deps → run `pytest tests/test_model_quality.py` → run `pytest tests/test_api.py` (basic endpoint smoke test).
   - If the quality test fails, the whole workflow fails — GitHub shows a red X, which is your "blocked deployment" evidence for screenshots.
5. Write `tests/test_api.py`: spins up the FastAPI app with `TestClient`, hits `/health` and `/predict` with a known payload, asserts response shape.

### Acceptance check
- Push a commit that intentionally breaks something (e.g., shrink training data drastically) → confirm GitHub Actions shows a failing red build.
- Push a fix → confirm it goes green.
- Screenshot both states now — you'll want them for your CV/portfolio writeup later, and re-creating this on demand during an interview is way harder than having it saved.

---

## 4. Week 3 — Drift Detection with Evidently AI

### Goals
- A script that simulates production traffic with intentionally shifted feature distributions.
- Evidently AI generates a drift report comparing that traffic against the Week 1 reference dataset.
- A threshold check that flags "drift detected: yes/no" programmatically (not just a pretty HTML report — you need a machine-readable signal to trigger Week 4's retraining).

### Tasks for the agent
1. `monitoring/simulate_traffic.py`:
   - Loads the held-out "future" data slice from Section 1.
   - Applies a deliberate perturbation to a few features (e.g., shift the mean of `transaction_amount` by 2 standard deviations, or resample from a different subpopulation) to simulate real drift instead of faking it with random noise — noise isn't drift, a distribution shift is.
   - Sends each row as a `POST /predict` request to your running API, with a small delay between requests to look like a traffic stream.
2. `monitoring/drift_check.py`:
   - Uses **Evidently AI's `DataDriftPreset`** (check current Evidently API — package has changed its interface across versions, confirm against the installed version's docs rather than assuming the interface from older tutorials).
   - Compares `data/reference/reference_data.csv` against a rolling window of recent requests pulled from `logs/predictions.jsonl`.
   - Outputs both: (a) an HTML report you can screenshot, and (b) a JSON summary with a boolean `drift_detected` and the specific drifted columns + statistical test used (e.g., Wasserstein distance, PSI, KS test — Evidently picks per-column defaults, know which ones fired so you can explain it if asked).
3. Decide and document your drift threshold explicitly (e.g., "drift detected if >30% of monitored columns show significant drift at p<0.05") — an arbitrary undocumented threshold is a red flag in review; a stated, justified one is a green flag.

### Acceptance check
- Run `simulate_traffic.py`, then `drift_check.py`.
- Confirm the JSON output correctly reports `drift_detected: true` when you've deliberately shifted the data, and `false` on a control run using undrifted data. Test both — a drift detector that always says "drift" is not a drift detector.

---

## 5. Week 4 — Alerting + Retrain Loop + Load Test Metrics

### Goals
- Drift detection triggers an alert and (optionally) an automated retrain that registers `v2` in MLflow.
- You generate the real latency/throughput numbers for your CV bullets — not invented ones.

### Tasks for the agent
1. `monitoring/retrain_trigger.py`:
   - Reads the `drift_check.py` JSON output.
   - If `drift_detected: true` → logs a clear console alert (timestamp, which columns drifted, magnitude) and optionally POSTs to a mock webhook (a free `webhook.site` URL is fine for a demo, or a local Slack incoming webhook if you have a workspace).
   - Triggers `training/train.py` again with the newer data included, which logs a new MLflow run and registers it as the next model version.
   - Prints a clear "rollback available: model v1 still registered, current serving: v2" message — this is what lets you truthfully say the system "prepares for rollback."
2. Load testing for real numbers (don't skip this — invented numbers on a CV are the difference between a strong project and a lie):
   - Use **Locust** (Python-native, fits your stack better than JMeter here).
   - Write a `locustfile.py` hitting `/predict` with realistic payloads.
   - Run against your Dockerized API locally, note actual p50/p95 latency and requests/sec your machine sustains.
   - **Use whatever numbers you actually measure on your CV, not the example numbers in this doc.** Your laptop's numbers are yours to claim; the "45ms / 500 req/s" figures floating around in generic MLOps tutorial content are not measurements you made, and claiming them is a fabrication that falls apart the moment an interviewer asks "how did you measure that." Run the test, screenshot the Locust results page, and use those exact numbers.
3. `monitoring/requirements.txt`: pin `evidently`, `locust`, `requests`.

### Acceptance check
- End-to-end demo run: start stack → run simulate_traffic → run drift_check → see it detect drift → run retrain_trigger → confirm MLflow UI (`localhost:5000`) shows a new registered `v2` model run with fresh metrics.
- Locust run completed with a saved report/screenshot.

---

## 6. README — What It Must Contain

Have the agent draft this last, once the code exists (a README written before the code tends to describe aspirations, not reality — write it against what actually runs):

- Architecture diagram (even ASCII is fine) showing: training → MLflow registry → FastAPI serving → traffic simulator → Evidently drift check → retrain loop.
- Exact `docker compose up` instructions — a reviewer/interviewer should be able to clone and run this in under 5 minutes with zero manual steps.
- Your **actual measured** metrics from Week 4, with the command used to reproduce them.
- The explicit, stated drift threshold and gate threshold decisions (Sections 1, 3) — showing your reasoning, not just your code, is what separates a real MLOps understanding from a tutorial-follow.
- A short "Known Limitations" section (e.g., "reference dataset is static; a production system would need periodic reference refresh" or similar) — including this voluntarily reads as engineering maturity, not weakness.

---

## 7. CV Bullet Template (fill in only with numbers you actually produced)

```
Automated MLOps CI/CD & Drift Monitoring Pipeline | Python, FastAPI, Docker, MLflow, GitHub Actions, Evidently AI
[Month Year] – Present

- Engineered an end-to-end GitOps pipeline for a fraud-detection model with automated
  CI/CD quality gates, blocking deployment when [your metric] fell below [your threshold]
- Built a drift-monitoring layer with Evidently AI comparing simulated live traffic
  against a reference distribution, detecting [N] drifted features using [statistical test(s) you saw fire]
- Closed the loop with automated retraining and MLflow Model Registry versioning,
  enabling rollback between model versions
- Load-tested the containerized inference API with Locust, measuring [your real p95 latency]
  and [your real req/sec] on [describe your test machine/setup briefly]
```

---

## 8. Pitfalls to Watch For (things that make this look like a copy-pasted tutorial instead of understood work)

- Gating CI/CD on raw "accuracy" for an imbalanced fraud dataset — use F1/recall on the minority class instead, and say why.
- Inventing load-test numbers instead of running Locust and reporting what actually came out.
- Calling random noise "drift" — real drift is a distribution shift, simulate it as one.
- Skipping the MLflow **Model Registry** and only using experiment tracking — versioning claims need the registry, not just logged runs.
- Not documenting your thresholds anywhere — arbitrary unstated numbers invite the obvious interview question "why 85%?" with no good answer.
