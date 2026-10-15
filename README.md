# EnergyMind Phase 1

Anomaly detection and classification on a smart electrical micro-grid:
20 houses, 4 zones, 1 transformer, measurements every 15 minutes over 240 days.

## 1. Overview

The system receives meter readings and produces, for each house and each time
step, a verdict among eleven classes (normal operation or one of the ten
anomalies) together with a calibrated confidence.

Three components with strictly separated roles:

| Component | Role | What it does not do |
|---|---|---|
| *LightGBM* | classifies known anomalies | — |
| *Isolation Forest* | atypicality / novelty score | it classifies nothing and invents no class name |
| *Physical rules* | consistency between house ↔ neighbors ↔ zone ↔ transformer | it learns nothing |

Guiding principle of the rule engine: *no sensor is an absolute truth*. Three
mutually independent aggregates are compared (the sum of the house meters, the
zone meter, and the transformer meter), and it is their disagreement that
points to the liar.

## 2. Final architecture

raw measurements
      │
      ▼
house + zone + transformer assembly
      │
      ▼
   24 features
      │
      ├──────────────┬────────────────────┐
      ▼              ▼                    ▼
  LightGBM     Isolation Forest     physical rules
      │              │                    │
      ▼              │                    │
 Vector Scaling      │                    │
      │              │                    │
      └──────────────┴────────────────────┘
                     │
                   fusion
                     │
                     ▼
        verdict + calibrated confidence
                     │
          confidence ≥ 0.80 ?  ──no──▶  UNCERTAIN
                     │ yes
                     ▼
               class diagnosis

*Fusion logic*, deliberately simple and documented:

- LightGBM says "normal" but the evidence signals an anomaly, and its best
  abnormal class exceeds promote_min_prob → promotion;
- LightGBM says "abnormal" without evidence and with a confidence below
  demote_max_prob → back to "normal".

Both thresholds are chosen by exhaustive search *on the validation set only*.

## 3. Dataset

Frozen 240-day dataset produced by the EnergyMind simulator (separate project).

| File | Rows | Content |
|---|---|---|
| data/raw_houses.csv | 460,800 | timestamp, house_id, zone_id, V, I, P, PF, E |
| data/raw_zones.csv | 92,160 | timestamp, zone_id, zone_P, zone_V |
| data/raw_transformer.csv | 23,040 | timestamp, trans_P |
| data/labels.csv | 576,000 | supervised target, three entity levels |

ground_truth.csv and injection_log.csv are *never read*. The loader
rejects any file whose schema contains an unexpected column, which would block
any leakage of simulation truth.

labels.csv provides label (the target) and event_id. **event_id is used
exclusively** to build leak-free splits and for event-level evaluation; it
never enters the feature matrix.

### Temporal splits

Never random row shuffling. One primary split plus four rolling splits, to
measure stability:

| Name | Train | Validation | Test |
|---|---|---|---|
| PRIMARY | d1–160 | d161–200 | d201–240 |
| S1 | d1–120 | d121–140 | d141–180 |
| S2 | d1–140 | d141–160 | d161–200 |
| S3 | d1–160 | d161–180 | d181–220 |
| S4 | d1–180 | d181–200 | d201–240 |

If an event overlaps a boundary, *all its rows are discarded* from that
split: the same event can never appear on both sides.

## 4. The 24 features

All derived from the observable measurements only, and all causal (current and
past steps only).

*12 original features* — src/features/base_features.py

| Feature | Measure |
|---|---|
| res_zone | house ↔ zone meter inconsistency, *technical loss subtracted* |
| res_trans | zones ↔ transformer meter inconsistency, loss subtracted |
| res_energy | (E(t) − E(t−1)) − P × 0.25, never the raw cumulative index |
| ratio_voisins | the house versus the other houses of its zone, after normalization |
| nb_voisins_anormaux | number of neighbors deviating at the same instant |
| variation | ΔP normalized by the house's baseline |
| frozen_count | consecutive nearly identical steps, *with tolerance* |
| rolling_std_6h | variability over the past 6 hours |
| drift_slope | drift slope over 12 h, measured on the gap to the neighbors |
| dev_tension | (V − 230) / 230 |
| trans_load_pct | transformer load as a % of nominal |
| res_VI | consistency P ≈ V × I × PF / 1000 |

*4 deviations from the house's own history* — src/features/baseline_features.py

deviation_from_house_baseline, relative_deviation_from_baseline,
zscore_vs_house_baseline, ratio_to_hour_median.

They compare the house to *its own history* rather than to its neighbors:
this is what distinguishes a meter offset from a zone-wide variation.

*8 persistence features* — src/features/persistence_features.py

mean_deviation_3h/6h/12h/24h, median_res_zone_6h/24h,
persistent_neighbor_deviation, consecutive_abnormal_count.

An appliance start lasts 1 to 4 steps, a meter bias 24 to 72: averaging over
several hours erases the former and keeps the latter.

### Learned statistics

| Statistic | Origin |
|---|---|
| per-house / per-slot / per-hour baselines | *train* medians |
| robust scales (median, MAD) | *train* |
| technical loss model (affine in load) | regression on the *train*, deciles 10–90 |
| frozen_count tolerance | 2% quantile of the *train* relative variations |
| "abnormal neighbor" threshold | 99% quantile of the *train* |
| rule thresholds | 99.5% quantile on the *clean* train instants |

No arbitrary constants. The feature builder is *re-fitted for each split*.

### Missing values

NaNs are *kept*: they carry information (missing_data class) and LightGBM
handles them natively. Isolation Forest, which does not accept them, receives a
copy imputed with the *train medians*.

## 5. Project structure

EnergyMind-Phase1/
├── data/                        frozen 240-day dataset
├── models/                      trained model (joblib)
├── src/
│   ├── config.py                loads config.yaml
│   ├── data/
│   │   ├── loader.py            CSV loading + inspection
│   │   ├── merge.py             house + zone + transformer assembly
│   │   └── splits.py            splits, leak protection
│   ├── features/
│   │   ├── windows.py           causal window primitives
│   │   ├── base_features.py     the 12 original features
│   │   ├── baseline_features.py the 4 history-based features
│   │   ├── persistence_features.py  the 8 persistence features
│   │   └── feature_pipeline.py  FeatureBuilder, FINAL_FEATURES
│   ├── models/
│   │   ├── lightgbm_model.py    multiclass classification
│   │   ├── isolation_forest.py  novelty score
│   │   └── calibration.py       Vector Scaling, ECE, abstention
│   ├── rules/physical_rules.py  consistency between levels
│   ├── inference/pipeline.py    preparation, fusion, confidence policy
│   └── evaluation/
│       ├── metrics.py           row-level and event-level metrics
│       └── plots.py             figures
├── outputs/{metrics,figures,predictions,datasets}/
├── tests/
├── config.yaml                  all parameters
├── main.py                      training + evaluation
├── run_calibration.py           confidence calibration
└── requirements.txt

## 6. Installation

pip install -r requirements.txt

**Environment note.** `scikit-learn` is pinned to **1.5.2**: on some Windows
machines, version 1.9.x is blocked by the application control policy
(`DLL load failed ... _expected_mutual_info_fast`), which makes
`sklearn.ensemble` unusable.


## 7. Running

python main.py

Trains and evaluates the model on the 5 splits (~45 min), and writes metrics,
figures, predictions and the saved model.

python main.py --quick --no-figures

Primary split only, for fast tuning.

python run_calibration.py

Confidence calibration. With --reuse, it reuses the probabilities already
saved instead of retraining.

python -m pytest tests -q

27 tests: loading, assembly, 24 features, component roles, calibration,
confidence policy.

## 8. Confidence calibration

Four methods are compared (none, temperature, vector, isotonic), all fitted
*on the validation set only* and selected by validation log-likelihood.

*Vector Scaling* is retained. Measured effect on the rows predicted abnormal:

| | Before | After |
|---|---|---|
| ECE | 0.1040 | *0.0453* |
| Actual precision | 84.7% | *90.6%* |

A telling example: the system used to announce sensor_drift with 77%
confidence while being right only 37% of the time. It now announces 66% for a
54% success rate.

*Abstention policy*: below inference.confidence_threshold (0.80), the
system answers *UNCERTAIN* instead of a class name. This keeps 89% of the
alerts with *96% precision*.

## 9. Results

Mean ± standard deviation over the 5 temporal splits.

| Metric | Value |
|---|---|
| Macro F1 | 0.830 ± 0.009 |
| Macro Recall | 0.830 |
| Anomaly recall | 0.804 |
| *False positive rate* | *0.60%* |
| Event-level recall | 98.5% |
| Accuracy | 97.2% |

Accuracy is deliberately secondary: 92% of the rows are normal, so a mute
model would already show 92%.

*F1 per class*: house_outage and zone_outage 1.00 · missing_data 0.98 ·
voltage_sag 0.96 · sensor_frozen 0.96 · overload 0.97 ·
voltage_spike 0.91 · falsified_reading 0.82 · sensor_bias 0.33 ·
sensor_drift 0.21.

## 10. Known limitations

- **sensor_bias and sensor_drift remain weak.** The system detects the
  problem in 67 to 77% of cases but often confuses the two classes with each
  other: they genuinely overlap. Four families of features were tested; only
  two held up across the 5 splits, and none improved drift.
- **sensor_bias is still 27 points overconfident** after calibration.
  Calibration makes the optimism visible, it does not remove it.
- *Detection delay*: a drift is detected after 26 steps at the median
  (6 h 30), and up to 57 steps at the 90th percentile.
- *A single simulated dataset.* The conclusions hold for this network and
  this simulator; a real network will require revalidation.
- The gap between training F1 (0.99) and test F1 (0.83) is large: the model
  learns the training set almost perfectly. It does not widen with the added
  features, but it remains to be monitored.

## 11. Reproducibility

- Fixed seed (model.random_state: 42), propagated to LightGBM and to
  Isolation Forest.
- No absolute paths in the modules: the project root is deduced from the
  location of src/config.py.
- All parameters are in config.yaml.
- The trained model is saved in models/, and the calibration probabilities
  in outputs/datasets/, so the analysis can be replayed without retraining.
