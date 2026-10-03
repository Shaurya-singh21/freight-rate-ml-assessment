# Freight Rate Prediction

Predicts the posted freight rate (in dollars) for freight loads. Built for the Spotter.ai Machine Learning Engineer assessment.

- Development data: `data/train-test.csv` — 48,000 labeled loads from January to October 2025
- Prediction data: `data/validation.csv` — 12,000 loads requiring final predictions
- Model: `HistGradientBoostingRegressor` predicting rate per mile, then multiplied by distance
- Validation: chronological, never random

## Results

All reported results come from `freight_rate_modeling.ipynb`.

"Clean" rows are defined using the RPM-based rule described in Data Quality. "All" rows include every row.

### Rolling monthly validation

Train on all months before the test month.

| Test month | Ridge clean MAE | Ridge clean MAPE | Boosting clean MAE | Boosting clean MAPE |
|---|---:|---:|---:|---:|
| 2025-07 | 76.4 | 3.18% | 47.2 | 1.86% |
| 2025-08 | 51.0 | 2.31% | 114.2 | 5.91% |
| 2025-09 | 51.4 | 2.05% | 30.8 | 1.21% |
| 2025-10 | 49.4 | 2.13% | 34.6 | 1.42% |
| **Average** | **57.1** | **2.42%** | **56.7** | **2.60%** |

Boosting has lower clean MAE in three of four months, including September and October. Ridge performs better in August. Across the four-month average, the two models are very close.

### Single chronological holdout

Train on January-August and test on September-October.

| Model | MAE (all) | Clean MAE | Clean MAPE |
|---|---:|---:|---:|
| Ridge baseline | 106.6 | 50.7 | 2.09% |
| Boosting without `market_index` (final) | 88.3 | 32.2 | 1.28% |
| Boosting with `market_index` | 134.7 | 79.2 | 3.11% |

RMSE and R² are reported in the notebook but are not used for model comparison because anomalous rate observations have a disproportionately large effect on squared-error metrics.

## Approach

### Key findings from exploration

- Rate per mile (`rpm = posted_rate / distance`) is a natural target representation. It decreases with distance and is higher for Flatbed and Reefer than for Dry Van.
- `quote_signal` has a curved relationship with RPM, so both the original feature and its square are used.
- `market_index` changes over time, but its relationship with rate is not stable across validation periods.
- Each city has a fixed coordinate pair, so city names and coordinates contain overlapping information.

### Data Quality

| Issue | What was found | Handling |
|---|---|---|
| Negative weights | About 290 rows have negative weights | `abs(weight)` |
| Capped weights | About 1,200 rows have weight exactly 47,500 | `weight_capped` flag |
| Missing values | 300 missing `weight`, 374 missing `market_index` | Median imputation |
| Anomalous rates | About 1.4% of rows have RPM far from the typical value for their distance range | Keep rows and use absolute-error loss |

A row is classified as clean when its RPM is between 0.55 and 1.6 times the median RPM for its distance range.

Training only on clean rows did not improve validation performance, so the final model uses all development rows.

### Features

The final model uses:

- pickup and delivery coordinates
- distance and log distance
- absolute weight
- `weight_capped`
- `quote_signal`
- `quote_signal` squared
- day of week
- pickup, delivery, and equipment as one-hot encoded categorical features

Excluded:

- **Month and day of year:** chronological validation showed that these features did not generalize well to future months.
- **`market_index`:** it improved some earlier validation periods but hurt the September-October holdout and was unstable across months.

### Model

`HistGradientBoostingRegressor` predicts RPM, which is then multiplied by distance.

| Setting | Value |
|---|---|
| loss | `absolute_error` |
| max_iter | 250 |
| learning_rate | 0.06 |
| max_leaf_nodes | 15 |
| l2_regularization | 1.0 |
| random_state | 42 |

The model was chosen because freight rate has a natural distance × RPM structure, several relationships are non-linear, and the data contains anomalous observations. Absolute-error loss provides robustness to these observations.

Ridge regression on log rate is used as the baseline.

### Validation and Split

All validation is chronological to avoid using future observations to predict earlier periods.

- Single holdout: January-August → September-October
- Rolling validation: July-October, training on all preceding months
- Final model: refit on all January-October data and used to predict `data/validation.csv`

## December Chart

The scorer fixes the route and load characteristics:

- Lexington → Fort Wayne
- 360 miles
- Dry Van
- 32,000 lb

Only the date changes.

The December input does not contain `quote_signal`, so the Dry Van training median is used. Since the final model does not use month or day-of-year, the December predictions mainly reflect the day-of-week effect.

Generated chart:

`scorer_results/candidate_december.png`

## Repository Layout

```text
.
├── data/
│   ├── train-test.csv
│   ├── validation.csv
│   ├── validation-predictions-template.csv
│   └── december-chart-inputs.csv
├── scorer_results/
│   └── candidate_december.png
├── freight_rate_modeling.ipynb
├── score.py
├── requirements.txt
├── validation_predictions.csv
├── december_predictions.csv
└── README.md
```

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the local data path in `.env`:

```text
MAIN_PATH=data
```

Run the notebook, then validate the outputs:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The scorer validates 12,000 final predictions and 31 December predictions and generates `scorer_results/candidate_december.png`.

## Limitations

- November and December target values are unavailable, so final test performance is unknown.
- The model does not explicitly capture month-level shifts in freight rates.
- Boosting performed substantially worse than Ridge in August, with 5.91% versus 2.31% clean MAPE.
- Final accuracy on the 12,000 validation loads will be measured by Spotter after submission.
