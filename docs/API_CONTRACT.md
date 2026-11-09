# Mastitis API contract (v1.2)

The single source of truth for the API, the dashboard and the SMS service.
The code version of this contract is [`api/schemas.py`](../api/schemas.py). The live OpenAPI docs are at
`http://<host>:8000/docs`. Example files: [`examples/predict_request.json`](../examples/predict_request.json)
and [`examples/predict_response.json`](../examples/predict_response.json) (the response example is real API
output for the request example).

> The model is trained on **SIMULATED** data (see README). Scores are for a prototype.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Risk for one milking. Also stores the reading in the cow's history. |
| `GET` | `/health` | `{"status": "ok", "model_loaded": true, "model_version": "rf-smote-v2"}` |
| `GET` | `/model-info` | Model name, version, threshold, risk bands, alert rule, test metrics |
| `GET` | `/history/{cow_id}?limit=50` | Stored readings for a cow, newest first (includes `cmt_score`, `risk_score`, `risk_level`, `alert`, `top_reasons`) |

CORS is open (`*`) for the prototype.

## POST /predict

### Request

```json
{
  "cow_id": "COW-012",
  "farm_id": "FARM-01",
  "timestamp": "2026-11-08T07:30:00",
  "ec_ms_cm": 6.8,
  "ec_quarters": {"LF": 5.4, "RF": 5.5, "LR": 7.9, "RR": 5.6},
  "milk_temp_c": 38.4,
  "ph": 6.95,
  "milk_yield_l": 6.2,
  "cmt_score": null,
  "breed": "exotic",
  "parity": 3,
  "days_in_milk": 120,
  "grazing_system": "zero"
}
```

| Field | Type | Required | Unit / allowed values | Meaning |
|---|---|---|---|---|
| `cow_id` | string (1-64) | yes | | Unique cow identifier. History, baselines and alerts are kept per `cow_id`. |
| `farm_id` | string \| null | no | | Farm identifier (stored only). |
| `timestamp` | ISO 8601 datetime \| null | no | farm local time | When the cow was milked. If omitted, server time is used. Send it whenever possible: history and alerts depend on it. |
| `ec_ms_cm` | number \| null | no | mS/cm, 0-20 | Whole-milk electrical conductivity. Used **only** when all four quarter readings are missing. |
| `ec_quarters` | object \| null | no | | Conductivity per quarter. **The main signal:** an infected quarter stands out from the others. |
| `ec_quarters.LF` / `RF` / `LR` / `RR` | number \| null | no | mS/cm, 0-20 | Left front, right front, left rear, right rear. Any quarter may be null. |
| `milk_temp_c` | number \| null | no | °C, 30-45 | Milk temperature at milking. |
| `ph` | number \| null | no | pH, 5-9 | Milk pH. Healthy milk is 6.6-6.8. |
| `milk_yield_l` | number \| null | no | litres, 0-60 | Yield at this milking. |
| `cmt_score` | integer \| null | no | 0-4 | California Mastitis Test on the 0-4 scale (= field scale 1-5 minus 1). **Not used for the prediction.** It is stored with the reading as a future training label (positive = 2 or more). |
| `breed` | string | yes | `exotic` \| `local` | Exotic = Friesian / crosses; local = Ankole / Zebu. |
| `parity` | integer | yes | 1-15 | Number of calvings. |
| `days_in_milk` | integer \| null | yes, unless known | 0-700 | Days since the last calving. May be omitted for a cow whose calving date is on record: it is then calculated from `cows.calving_date`. Whenever it *is* sent, the calving date is updated from it. Omitted for a cow with no calving date → 422. |
| `grazing_system` | string | yes | `zero` \| `tethered` \| `open` | Housing / grazing system. |

**Missing sensor values (sensor fault).** Send `null` or leave the field out. The API never fails because
of missing sensor data. It falls back the same way as training (`ml/features.py`):
- Quarter EC: uses the quarters that are present. If all four are missing, it uses `ec_ms_cm`. If that is
  missing too, it uses the cow's 7-day average, or the herd median for her breed.
- pH / temperature / yield: uses the cow's previous reading. If there isn't one, it uses the herd median.
- Accuracy is lower when key readings (especially quarter EC) are missing.

### Response (200)

```json
{
  "cow_id": "COW-012",
  "risk_score": 0.98,
  "risk_level": "high",
  "top_reasons": [
    {"feature": "ec_quarter_diff", "effect": "+0.30", "text": "Left rear quarter conductivity much higher than others (+2.4 mS/cm)"},
    {"feature": "ph", "effect": "+0.09", "text": "Milk pH higher than normal (6.95; normal is 6.6-6.8)"},
    {"feature": "ec_vs_baseline", "effect": "+0.05", "text": "Conductivity 14% above the herd average"}
  ],
  "advice": "Likely mastitis in the left rear quarter. Do a CMT test on this cow today and call your vet. Milk her last, and do not sell milk from a treated cow until the withdrawal period ends.",
  "model_version": "rf-smote-v2",
  "alert": false
}
```

| Field | Type | Meaning |
|---|---|---|
| `cow_id` | string | Echo of the request. |
| `risk_score` | number 0-1, 2 decimals | Model score; higher = more likely subclinical mastitis. It ranks cows; it is **not** a calibrated probability. |
| `risk_level` | `low` \| `medium` \| `high` | Band for **this reading** (see risk bands below). |
| `top_reasons` | array of 1-3 | Main reasons, most important first. Reasons with \|effect\| < 0.03 are left out (too small to matter), but there is always at least one. |
| `top_reasons[].feature` | string | The model feature behind the reason, e.g. `ec_quarter_diff`, `ec_vs_baseline`, `ph`. |
| `top_reasons[].effect` | string | Signed contribution to the score (SHAP), always 2 decimals: `"+0.21"`, `"-0.05"`. Closely related features (e.g. the three quarter-EC features) are combined into one reason. |
| `top_reasons[].text` | string | Plain-English sentence for the farmer. Quarters are named in words ("Left rear quarter ..."). |
| `advice` | string | What the farmer should do, by risk level (below). |
| `model_version` | string | From the saved model metadata, e.g. `rf-smote-v2` (model + imbalance method + version). |
| `alert` | boolean | `true` only when this reading **and** the cow's previous reading are both `high`. **Only `alert: true` should trigger an SMS.** |

**Reason direction.** For `medium`/`high` the reasons are the factors that *raise* risk. For `low` they are
reassuring factors ("Conductivity similar in all four quarters"). Breed, grazing system, parity and
lactation stage are only ever shown as factors that *raise* risk.

### Risk bands

`low < 0.25 <= medium < threshold <= high`, where `threshold` is tuned at training time for recall >= 0.85.

The current model (`rf-smote-v2`) has threshold **0.416**, so today: **low < 0.25 <= medium < 0.416 <= high**.
Medium means "worth watching": the score is raised but below the alert threshold.
(If a retrained model's threshold were ever <= 0.25, the medium band would be empty; the API would then
start medium at half the threshold instead.)
`GET /model-info` → `risk_bands` always returns the bands actually in use. Read them from there instead
of hard-coding numbers in the dashboard.

### Advice text by level

| Level | Advice |
|---|---|
| `low` | No sign of mastitis. Keep teats clean and dry, dip teats after milking, and do a CMT check every week. |
| `medium` | Watch this cow. Do a CMT test at the next milking, milk her last, and check her readings again tomorrow. |
| `high` | Likely mastitis in the *<quarter>* quarter. Do a CMT test on this cow today and call your vet. Milk her last, and do not sell milk from a treated cow until the withdrawal period ends. |

### Alert rule (SMS)

- `risk_level` is per reading and can be `high` because of one noisy reading (e.g. a dirty electrode).
- `alert: true` requires **two consecutive `high` readings** for the same `cow_id`. "Consecutive" means
  the cow's most recent stored reading before this `timestamp` was `high`, and it is **at most 48 h**
  older. Two highs several days apart are not a sudden pattern, so they don't alert.
- A normal reading in between resets it: high → low → high gives `alert: false`.
- While the cow stays high, every further reading also returns `alert: true`. If the SMS service should
  send only one message per episode, it should send on the first `true` after a `false`.
- Re-sending the same `cow_id` + `timestamp` replaces that reading (no duplicates, no double-counting).

## Errors

### 422: invalid input

Returned for an invalid enum value, an impossible number (pH outside 5-9, EC outside 0-20, temperature
outside 30-45 °C, yield outside 0-60 L, `cmt_score` outside 0-4, parity outside 1-15, days in milk
outside 0-700), a missing required field, or wrong types.

```json
{
  "error": "validation_error",
  "message": "ph: Input should be less than or equal to 9; breed: Input should be 'exotic' or 'local'",
  "detail": [
    {"type": "less_than_equal", "loc": ["body", "ph"], "msg": "Input should be less than or equal to 9", "input": 9.5, "ctx": {"le": 9.0}},
    {"type": "literal_error", "loc": ["body", "breed"], "msg": "Input should be 'exotic' or 'local'", "input": "unicorn", "ctx": {"expected": "'exotic' or 'local'"}}
  ]
}
```

- `message`: one human-readable line listing every problem as `field: reason` (nested fields as `ec_quarters.LF`).
- `detail`: the standard FastAPI/Pydantic per-field list, for programmatic handling.

### 503: model not loaded

`{"detail": "Model not loaded - run `python -m ml.train` first"}`. The server started without a trained
model in `models/`.

## Changelog

- **v1.2.1:** the medium band now starts at 0.25 (was 0.4): `low < 0.25 <= medium < threshold <= high`.
  Request and response fields are unchanged.

- **v1.2** (backward compatible, request and response otherwise unchanged): `days_in_milk` may be omitted for a
  cow whose calving date is on record. Every reading is now stored in the database (`data/app.db`). An
  `alert: true` also opens or extends an alert *episode* on the dashboard, and only a new episode sends an
  SMS (see the README). The `alert` field itself is unchanged.

- **v1.1:** this contract. Quarter keys `LF/RF/LR/RR` in `ec_quarters`; whole-milk `ec_ms_cm`; optional
  `cmt_score` stored as a label; `top_reasons` as `{feature, effect, text}`; new `alert` field; readable 422
  errors. The response no longer includes `timestamp`, `threshold`, `suspect_quarter` or
  `history_readings_used`. Use `/model-info` for the threshold and `/history/{cow_id}` for stored readings.
