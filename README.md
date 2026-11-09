# MastitisAI: early warning for subclinical mastitis (university prototype)

Early detection of **subclinical mastitis (SCM)** in dairy cows for smallholder farmers in Uganda,
using in-line milk sensors: electrical conductivity (EC) per quarter, pH, milk temperature and yield.
One FastAPI app serves the prediction API, a farmer/vet web dashboard, SMS alerts and a USSD menu.

> ⚠️ **ALL DATA IN THIS REPOSITORY IS SIMULATED.**
> `data/synthetic_milkings.csv` is produced by `ml/generate_synthetic.py`. No real farm,
> cow or sensor data has been used. All results below describe performance on **simulated**
> data only, and they will be optimistic compared with real field data.

## Architecture

Three layers, each depending only on the ones below it:

```
api/  HTTP layer      routes parse input, call app/, return the result (no business logic)
app/  domain layer    database, reading pipeline, alert episodes, SMS, dashboard data, USSD menu
ml/   ML layer        synthetic data, features, training, evaluation, SHAP
      app/config.py   shared constants (risk bands, alert rules, quarter names, valid ranges),
                      imported by all three layers; it imports nothing from the project
```

```
api/
  main.py                FastAPI app: middleware, static files, error handler, routers
  schemas.py             request/response contract (Pydantic) - human version: docs/API_CONTRACT.md
  errors.py              the 422 error format
  routes/predict.py      POST /predict, GET /health, GET /model-info, GET /history/{cow_id}
  routes/dashboard.py    pages /, /cows/{id}, /alerts, /cmt, /model, /demo, /sms + JSON /api/...
  routes/ussd.py         POST /ussd (Africa's Talking callback)
app/
  config.py              shared constants (one place for every rule and limit)
  db.py, models.py       SQLite via SQLAlchemy: farms, cows, readings, alerts, sms_log
  service.py             handle_reading(): the reading pipeline (+ data for /history, /model-info)
  predictor.py           loads the model once, scores one reading, risk bands
  reasons.py             SHAP values -> plain-English reasons + advice
  alerts.py              alert episodes: open / attach / auto-resolve / farmer close / re-arm
  notify.py              send_sms() (console | Africa's Talking), 12 h per-cow limit, sms_log
  sms_templates.py       SMS texts by language ("en"; "lg" Luganda placeholder)
  dashboard.py           queries and page data for the dashboard
  ussd.py                USSD menu logic
  replay.py              synthetic CSV row -> /predict request (seed, demo, simulator)
  seeding.py             seed the database with 30 days of replayed history + farmer actions
  demo.py                "Start live demo" background replay
ml/
  generate_synthetic.py  synthetic dataset generator (all parameters in CONFIG)
  features.py            build_features(): the ONE feature function used by training AND the API
  episodes.py            the alert/episode rules replayed over history (clean-baseline flags)
  train.py               farm-level split, stage-0 flags, 3 models x 2 imbalance strategies, CV, threshold
  evaluate.py            test-farm metrics, confusion matrix, ROC curves, SHAP, metrics.json
  explain.py             SHAP helper shared by evaluate.py and the API
scripts/seed_db.py       command line for app/seeding.py
simulator/send_readings.py  fake sensor that POSTs to /predict (terminal demo)
templates/, static/      Jinja2 + Bootstrap 5 + Chart.js (CDN), no build step
models/, reports/        trained model + metadata; comparison tables, plots, metrics.json
docs/API_CONTRACT.md     API contract: the single source of truth for API, dashboard and SMS
examples/                predict_request.json / predict_response.json
tests/                   pytest
pyproject.toml           ruff (lint + format) and pytest configuration
```

**How a reading flows through the system.** A sensor (or `simulator/send_readings.py`) sends one
milking to **`POST /predict`**. `api/routes/predict.py` validates it against `api/schemas.py` and calls
`app/service.handle_reading()`. The service finds or creates the cow, loads her last 22 days of readings
from SQLite and passes history + new reading to **`ml/features.build_features()`**, the same function used
in training, which computes quarter conductivity differences and the clean baseline. **The model**
(`app/predictor.py`) returns a risk score and SHAP values. The score is mapped to low/medium/high with
the bands in `app/config.py`, and **`app/reasons.py`** turns the SHAP values into up to three
plain-English **reasons** and advice. The reading is saved, and **`app/alerts.py`** applies the episode
rules: two high readings in a row open an **alert episode**, later highs join it, and four normal
readings close it. Only a new episode calls **`app/notify.py`**, which sends one **SMS** (console or
Africa's Talking) and logs it. The **dashboard** (`api/routes/dashboard.py` + `app/dashboard.py`) reads the
same database to show the herd, each cow's history and reasons, open alerts and the SMS log. There,
farmers mark alerts checked / treated / false alarm and enter CMT results that become future training labels.

## How to run

```powershell
# 0. setup (once)
python -m venv venv
venv\Scripts\activate              # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env             # Linux/macOS: cp .env.example .env  (defaults work as-is)

# 1-3. ML pipeline (already done if models/ and reports/ exist)
python -m ml.generate_synthetic   # SIMULATED data -> data/synthetic_milkings.csv
python -m ml.train                 # -> models/
python -m ml.evaluate              # -> reports/

# 4. seed the dashboard database (3 farms x 10 cows, last 30 days; ~2-6 min)
python scripts/seed_db.py          # add --reset to wipe data/app.db and seed again

# 5. run the app
uvicorn api.main:app --reload
#    dashboard  http://127.0.0.1:8000/
#    API docs   http://127.0.0.1:8000/docs

# tests, lint and formatting
pytest
ruff check .
ruff format .
```

### .env

| Variable | Default | Meaning |
|---|---|---|
| `SMS_PROVIDER` | `console` | `console` prints SMS in the server terminal (nothing is sent). `africastalking` sends through Africa's Talking. |
| `AT_USERNAME` | `sandbox` | Africa's Talking username (`sandbox` for testing) |
| `AT_API_KEY` | - | Africa's Talking API key (sandbox key from the AT dashboard) |
| `AT_SENDER_ID` | empty | Optional sender ID / short code |
| `SMS_LANGUAGE` | `en` | `en`, or `lg` (Luganda placeholder that falls back to English) |
| `MASTITIS_DB` | `data/app.db` | SQLite database file |

`.env` and `data/app.db` are git-ignored. Seeding always uses the console SMS provider unless
you pass `--send-sms`, so 30 days of history never sends real messages.

### Demo steps (for the panel)

1. `python scripts/seed_db.py`, then `uvicorn api.main:app`, and open http://127.0.0.1:8000/.
2. **Herd** (`/`): cows sorted by latest risk, with red/amber/green badges. Filter by farm and click a cow.
3. **Cow detail**: risk over time with the threshold line, EC per quarter (the infected quarter rises
   above the others), pH and temperature, the plain-English reasons, and the alert episodes.
4. **Live demo** (`/demo`): press *Start live demo*. Three cows from the held-out test farms are replayed,
   one milking every 2 s. One cow develops mastitis: its risk turns red, an alert opens after the
   **second** high reading, and exactly **one** SMS appears in the server terminal and on `/sms`.
5. **Alerts** (`/alerts`): mark the alert *Checked / Treated / False alarm* with a note.
6. **CMT result** (`/cmt`): enter a CMT score (1-5 field scale). It is stored as a training label.
7. **Model** (`/model`): version, test-farm metrics, confusion matrix and SHAP plot, with the simulated-data caveat.

### Screenshots

> Placeholders: add screenshots to `docs/screenshots/` before the defence.

| Page | Screenshot |
|---|---|
| Herd overview | `docs/screenshots/herd.png` |
| Cow detail | `docs/screenshots/cow.png` |
| Alerts | `docs/screenshots/alerts.png` |
| Live demo | `docs/screenshots/demo.png` |
| SMS log | `docs/screenshots/sms.png` |
| Phone view | `docs/screenshots/mobile.png` |

## 1. Synthetic data

300 cows on 10 farms, milked twice a day for 90 days (54,000 milkings).

| Aspect | Simulation |
|---|---|
| Cows | breed exotic 70% / local 30%, parity 1-7, days in milk 5-310, grazing system per farm (zero / tethered / open) |
| Prevalence | daily infection hazard calibrated so **~60% of cows** have >= 1 SCM episode (Ugandan studies: 53-90%). Achieved: 58% |
| Risk factors | relative risk x1.8 exotic, x1.5 parity > 1, x1.6 zero-grazing, x1.7 early lactation (< 60 DIM), plus a random farm hygiene effect |
| Episodes | 1 quarter (20%: 2 quarters), 7-30 days, ramping up over 2-4 days |
| Healthy milk | EC 4.5-6.0 mS/cm, pH 6.6-6.8, 37.0-38.0 °C |
| Infected | EC +1.5-3.5 mS/cm in the infected quarter, pH 6.9-7.2, temp +0.3-0.8 °C, yield -5-15% |
| Noise | sensor noise, per-farm sensor calibration offset, rare EC spikes (dirty electrode), ~3% missing values |
| Label | `cmt_score` 0-4 (= field CMT 1-5 minus 1), recorded on ~1 in 7 milkings, 5% human misreading. **Positive = cmt_score >= 2** (CMT >= 3) |

**One deliberate deviation from the brief:** in a first version every infection shifted pH and
temperature exactly in step with EC. The model then learned to *require* abnormal pH, so a
quarter at 8-9 mS/cm with normal pH scored only ~0.3 (low risk). That would be dangerous in
the field, where EC is the main in-line signal and the pH of mixed milk is a weaker,
inconsistent marker (one infected quarter is diluted by three healthy ones). Now 70% of
episodes show the full pH/temperature shift and the rest show only 0-30% of it
(`p_systemic_response` in CONFIG; set it to `1.0` to restore the original behaviour).

`true_infected` / `true_infected_quarters` in the CSV are ground truth for analysis and the
demo only. They are **never** used as model inputs.

## 2. Features (`ml/features.py`)

`ec_mean, ec_max, ec_quarter_diff` (highest quarter minus the mean of the others), `ec_quarter_ratio`,
`ec_vs_baseline / yield_vs_baseline / temp_vs_baseline` (% change vs the cow's own **clean baseline**,
see below), `ph, milk_temp_c, milk_yield_l, yield_change_3d, parity, days_in_milk`, one-hot `breed_*`
and `grazing_*`.

**Clean baseline (model v2).** A cow's "normal" is the mean of her previous **clean** readings:
the last 7 days if there are at least 3, otherwise the last 21 days if there are at least 3, otherwise
the herd median for her breed. A reading is *not clean* if it was high risk, or if an alert episode was
active at the time. Without this, a 3-week infection slowly becomes the cow's "normal" and
`ec_vs_baseline` drifts back to 0 while she is still sick.

* **Live service:** each reading's flag is stored in `readings.baseline_exclude` when it is processed
  (`app/service.py`), from the model's own risk level and the episode state.
* **Training** needs the same flag for every historical reading, and that needs a risk level. A
  **stage-0 model** (RandomForest, plain baseline) scores every reading **out-of-farm**: training
  farms by 5-fold GroupKFold, test farms by a model trained on the training farms. Its high/low
  calls are replayed through the same episode rules as the API (`ml/episodes.py`), and the final
  models are trained on the clean-baseline features. Every flag uses only the current and earlier
  readings, so there is no future leakage.
* Stage 0 excluded 29.9% of all readings: 99.5% of truly infected readings, and 18.6% of healthy ones.
  Its threshold is tuned for recall, so it is cautious, and every cow still keeps plenty of clean readings.

* **Missing values:** EC summaries use the quarters available; ph/temp/yield are carried forward from
  the cow's last reading, and anything still missing falls back to herd medians.
* **New cow (no history):** her baseline is the herd median for her breed, computed from the
  **training farms only** and saved with the model.

## 3. Training (`ml/train.py`)

* Only CMT-labelled milkings are used for training (7,770 of 54,000).
* **Split by farm:** 2 whole farms (F02, F09) are held out as the test set, so the model is
  judged on farms it has never seen.
* Class balance is printed first: train 13.3% positive, test 21.3% positive.
* 3 models (LogisticRegression baseline, RandomForest, XGBoost) x 2 imbalance strategies
  (`class_weight` / `scale_pos_weight` vs SMOTE inside an imblearn pipeline, so oversampling only
  ever sees training folds).
* 5-fold **GroupKFold by farm** on the training farms. Selection rule: highest recall with
  precision >= 0.5 (tie-break ROC-AUC).
* **Decision threshold** is tuned on out-of-fold predictions: the highest threshold that still
  gives recall >= 0.85.
* Before all of this, the stage-0 model creates the clean-baseline flags (see section 2).

## 4. Results (SIMULATED data)

**Selected model: RandomForest + SMOTE (`rf-smote-v2`, clean baseline)**, threshold 0.416.

Held-out farms F02 + F09 (1,543 labelled milkings, 328 positive):

| model | accuracy | precision | recall | F1 | ROC-AUC | specificity | threshold |
|---|---|---|---|---|---|---|---|
| LogReg + class_weight | 0.827 | 0.556 | 0.924 | 0.694 | 0.947 | 0.801 | 0.289 |
| LogReg + SMOTE | 0.809 | 0.529 | 0.927 | 0.673 | 0.947 | 0.777 | 0.277 |
| RandomForest + class_weight | 0.843 | 0.584 | 0.902 | 0.709 | 0.936 | 0.826 | 0.187 |
| **RandomForest + SMOTE** | **0.945** | **0.865** | **0.878** | **0.871** | **0.946** | **0.963** | 0.416 |
| XGBoost + scale_pos_weight | 0.756 | 0.462 | 0.902 | 0.611 | 0.927 | 0.716 | 0.117 |
| XGBoost + SMOTE | 0.931 | 0.811 | 0.878 | 0.843 | 0.938 | 0.945 | 0.192 |

Best model confusion matrix (test farms): TP 288, FN 40, FP 45, TN 1,170.

**Effect of the clean baseline** (selected model, held-out farms, v1 → v2):

| | v1 (plain 7-day baseline) | v2 (clean baseline) | change |
|---|---|---|---|
| Recall | 0.887 | 0.878 | -0.009 (3 more missed of 328) |
| Precision | 0.793 | 0.865 | +0.072 |
| Specificity | 0.937 | 0.963 | +0.026 |
| False alarms | 76 | 45 | -41% |
| ROC-AUC | 0.946 | 0.946 | 0 |

Details and the explanation: [reports/baseline_v1_vs_v2.md](reports/baseline_v1_vs_v2.md).
Overall *ranking* (ROC-AUC) is unchanged, but at the operating point the model raises far fewer false
alarms for almost the same recall. One train/test split on simulated data: treat ±0.01 differences
as noise.

See `reports/` for `test_comparison.*`, `cv_comparison.md`, `confusion_matrix.png`,
`roc_curves.png`, `shap_summary.png` and `metrics.json`.

**How to read this:**
* All models rank cows similarly well (ROC-AUC 0.93-0.95). The logistic regression baseline is
  almost as good at *ranking*, which means the problem is mostly carried by a few strong signals
  (the EC quarter difference first, then pH).
* The real difference is how well each **threshold** transfers to new farms. Thresholds tuned on the
  training farms gave the class-weighted models much lower precision on the test farms (0.48-0.61).
  The SMOTE models' probabilities transferred better.
* SHAP: `ec_quarter_diff`, `ec_quarter_ratio`, `ec_max` and `ph` drive predictions, which matches
  the biology.

## 5. API

**The contract is defined in [docs/API_CONTRACT.md](docs/API_CONTRACT.md).** It is the single source of
truth for the API, dashboard and SMS service. It covers every field, its unit and allowed range, the
risk bands, the alert rule and the error format. The code version is `api/schemas.py`, and the live
OpenAPI docs are at `/docs`. Ready-to-use examples: [examples/predict_request.json](examples/predict_request.json)
and [examples/predict_response.json](examples/predict_response.json).

Endpoints: `POST /predict`, `GET /health`, `GET /model-info`, `GET /history/{cow_id}`. CORS is open (`*`).

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @examples/predict_request.json
```

In short:
* **Quarter keys** are `LF/RF/LR/RR` in `ec_quarters`. Internally the API maps them to the
  generator/feature columns `ec_lf/ec_rf/ec_lr/ec_rr`.
* **Missing sensor values** (`null` or omitted) never crash the API. They fall back exactly like
  `ml/features.py`: other quarters → whole-milk `ec_ms_cm` → the cow's 7-day average → herd median.
* **`cmt_score`** (optional, 0-4) is stored with the reading as a future training label. It is not used
  for the prediction.
* **`risk_level`** is per reading: `low < 0.25 <= medium < threshold <= high`. The tuned threshold is
  0.416, so today: low < 0.25 <= medium < 0.416 <= high. `/model-info` → `risk_bands`.
* **`alert`** is `true` only for **2 consecutive `high` readings** of the same cow (at most 48 h apart).
  One noisy spike never alerts. Only `alert: true` should trigger an SMS.
* **`model_version`** comes from the saved model metadata (`rf-smote-v2` = RandomForest + SMOTE,
  version 2; set by `MODEL_VERSION` in `ml/train.py`).
* **`top_reasons`** leaves out reasons with |effect| < 0.03, but always returns at least one.
* **Storage:** every reading, with its risk score, level, alert flag, reasons and model version, is saved
  in `data/app.db` (SQLAlchemy). It is keyed on (cow, timestamp), so re-sending a milking replaces it.
* **`days_in_milk`** may be omitted for a cow whose calving date is on record (it's calculated). This is
  the only change to the contract (v1.2), and it is backward compatible.

## 6. Dashboard, alert episodes and SMS

**Database** (`app/models.py`): `farms`, `cows` (tag = the API's `cow_id`, with `calving_date`), `readings`
(sensor values + model output + optional CMT label), `alerts` (one row per **episode**) and `sms_log`
(every SMS attempt). All writes go through `app/service.py: handle_reading()`, the same code for
`/predict`, the seed script and the live demo.

**Alert episodes.** `/predict` returns `alert: true` on *every* reading while a cow stays high (2+ highs in a
row). The dashboard treats that run as **one episode**:

| Event | What happens |
|---|---|
| `alert: true`, cow has no active alert | new `alerts` row (status `open`) + **one SMS** |
| `alert: true`, cow already has an active alert | reading attached to it (`last_reading_id`, `reading_count += 1`), no SMS |
| 4 consecutive non-high readings while active | status `resolved_auto` (episode over) |
| Farmer marks `treated` or `false_alarm` | episode closed. No new episode while the cow is *still* high (antibiotics take days): it needs one non-high reading first ("re-arm"). If the cow goes high again after that, it's a **new** episode and a new SMS |
| Farmer marks `checked` | still active (acknowledged, not finished), so no new SMS while it continues |

`/alerts` and the cow page show each episode's length: first → latest reading, number of readings and hours.
False alarms are kept, not deleted, as negative examples for retraining.

**SMS** (`app/notify.py`): sent only when a new episode is created, to the farm's phone. As a safety net
there is at most one SMS per cow per 12 hours. Every attempt is logged in `sms_log` with status `sent`,
`failed`, `rate_limited` or `no_phone`, and shown on `/sms`. Messages fit one SMS (≤ 160 characters), e.g.
`MastitisAI: Cow F02-C005 (Farm Kayunga) HIGH risk. Right front quarter. Do CMT test today and call your vet.`
Texts live in `app/sms_templates.py` by language. The `lg` (Luganda) entries are placeholders that fall
back to English until translations are added.

**Pages:** `/` herd overview · `/cows/{id}` cow detail · `/alerts` · `/cmt` CMT entry · `/model` ·
`/demo` live demo · `/sms` SMS log. **JSON** used by the pages: `GET /api/cows[?farm=]`,
`GET /api/cows/{id}/readings[?days=30]`, `GET /api/alerts[?status=active|all]`,
`PATCH /api/alerts/{id}` with `{"status": "checked"|"treated"|"false_alarm", "note": "..."}`,
`POST /api/demo/start`, `GET /api/demo/status`.

**CMT entry** (`/cmt`): the score is entered on the familiar 1-5 field scale, stored as 0-4, and attached to
the cow's last recorded milking on that date. If there is no milking that day, it is refused with a message.

**USSD** (`app/ussd.py` + route `api/routes/ussd.py`, optional): Africa's Talking callback at `POST /ussd`. The farm is
identified by the caller's phone number. `1` My high-risk cows · `2` Enter CMT result (cow tag → score 1-5)
· `3` Mark alert as treated. Responses use the `CON` / `END` format. Point the AT sandbox USSD callback
URL at `https://<your-tunnel>/ussd` (e.g. via ngrok).

**Seed data** (`scripts/seed_db.py`): farms F02 and F09 (the held-out test farms) and F04, with 10 cows
each (about half with an infection in the window), last 30 simulated days, replayed through
`handle_reading()`. Timestamps are shifted so the history ends today. Farm names, owners and phone
numbers are fictional. Farmer behaviour is simulated as well. Each episode that starts more than
3 days before today gets a fate: ~60% `treated` ("Treated by vet, intramammary antibiotic"), ~15%
`false_alarm`, the rest left open. The farmer acts 12-36 h after the alert. A treated cow's remaining
infected readings are replaced by healthy ones (her own healthy levels + sensor noise) from 5-10 days
after treatment.

## 7. Terminal demo (simulator)

`simulator/send_readings.py` picks cows from the **held-out test farms**: one that develops mastitis
plus two that stay healthy. It replays them from ~8 days before the infection, so the
baseline builds up. In the recorded run the infected cow (F02-C005, infection starts day 33)
stayed low on days 25-32 and was flagged **high** at the second milking of day 34, about half a day
after onset, with the reason "Right front quarter conductivity much higher than others (+1.3 mS/cm)".
The **SMS alert** (`alert: true`, shown as `SMS!`) fires at the next reading (day 34 pm), when the cow
is high for the second time in a row. The two healthy cows each had one high reading caused by a
simulated electrode spike (days 27 and 41). Neither triggered an SMS: 15 alerts in the run, all on the
infected cow, none on healthy cows.

## Limitations (say these to the panel before they ask)

1. **Simulated data.** The model can only rediscover the rules we put into the generator, and
   real performance will be lower. Next step: collect real sensor + CMT data and retrain/re-validate.
2. **Few farms.** Grazing system is a farm-level attribute, and with only 8 training farms (one
   zero-grazing farm after the split) the model cannot learn its effect reliably. SHAP shows
   zero-grazing slightly *lowering* risk even though the data has it raising risk. That's why the
   API never presents breed or grazing as protective.
3. **Single-reading false alarms.** A healthy cow can get a `high` reading after one noisy EC reading
   (the simulator includes rare electrode spikes). This is handled by the `alert` rule: an SMS needs 2
   consecutive high readings, and only once per episode. The cost is that the SMS comes one milking
   (~12 h) later than the first high reading.
7. **Prototype security.** The dashboard and API have no login and CORS is open. Before real farm data
   is used, add authentication (at least per-farm access), HTTPS, and validation of the Africa's
   Talking USSD callback.
4. **SMOTE on one-hot columns** creates fractional values for synthetic samples. SMOTENC would be
   cleaner, but results were already good, so the simpler version was kept.
5. **Probabilities are not calibrated** (SMOTE shifts them). Treat the score as a ranking and
   rely on the tuned threshold, not on "97% chance".
6. **Per-milking metrics.** Rows from the same cow are correlated. Cow-level or episode-level
   detection delay would be a better field metric (the simulator shows it qualitatively).
