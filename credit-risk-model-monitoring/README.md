# Credit risk model performance and monitoring

A learning project on public credit data. It compares a logistic regression, a CART decision tree and a random forest, checks how well each ranks and prices credit risk on unseen cases, and builds a simple monitoring report with performance by segment and a simulated input shift.

## Objective

Assess model performance the way a monitoring analyst would: check data quality, evaluate on a held-out test set, separate ranking ability from probability accuracy, look at segments, and state clearly what the data can and cannot support.

## Data

[South German Credit (update)](https://archive.ics.uci.edu/dataset/573/south+german+credit), UCI Machine Learning Repository, 2020. DOI: [10.24432/C5QG88](https://doi.org/10.24432/C5QG88). Licence: CC BY 4.0. The source files are in `data/`, with the UCI code table.

- 1,000 historical consumer credits (1973-1975), 20 predictors, no missing values, no duplicate rows.
- Outcome: good or bad credit (source coding 1 = good, 0 = bad). The notebooks recode it to 1 = bad credit.
- Bad credits are over-sampled: 300 bad (30%) and 700 good.

## Structure

The notebooks are run in order. Shared code is in `notebooks/helpers.py`.

| Notebook | Purpose |
|---|---|
| `A_data_preparation` | Load, rename and decode the data; quality checks; class balance; exploratory plots |
| `B_data_preprocessing` | Stratified 70/30 train/test split; preprocessing design with no leakage from the test set |
| `C1_crm_LR` | Logistic regression: tuning, test-set PD, calibration, coefficients |
| `C2_crm_CART` | CART tree: tuning, test-set PD, calibration, the fitted tree |
| `C3_crm_RF` | Random forest: tuning, test-set PD, calibration, permutation importance |
| `D_comparison` | ROC-AUC, Gini, Brier score and calibration side by side |
| `E_monitoring` | Performance by segment, a labelled simulated input shift (PSI), monitoring summary |

## Method

- Preprocessing (median/most-frequent imputation, standardising, one-hot encoding) sits inside each model's pipeline, so it is fitted on training data only, also inside every cross-validation fold.
- Hyperparameters are tuned with 5-fold cross-validation repeated 3 times, scored by ROC-AUC, on the training set only. The test set is used once, to report performance.
- Performance is computed from predicted probabilities: ROC-AUC with a bootstrap 95% interval, Gini (2 x AUC - 1), Brier score, and calibration in equal-sized score bins.
- The data are not resampled, so predicted PDs can be compared with observed rates.
- The input shift is a **simulation** (loan durations x 1.25 in a copy of the test set), not observed deterioration.

## Results (test set: 300 cases, 90 bad)

| Model | ROC-AUC (95% interval) | Gini | Brier score |
|---|---|---|---|
| Logistic regression | 0.768 (0.710-0.824) | 0.535 | 0.169 |
| CART tree | 0.722 (0.662-0.775) | 0.444 | 0.183 |
| Random forest | 0.789 (0.733-0.842) | 0.579 | 0.167 |

- The random forest scores highest, but its interval overlaps heavily with the logistic regression's, so with this sample size there is no clear evidence that it ranks better. The tree, limited to a shallow depth, scores lowest.
- The logistic regression is nearly as good as the random forest and easier to explain, so it is the model monitored in notebook E.
- Its observed bad rate rises with loan duration (18%, 30%, 49%) and the predicted PD follows it. Segment AUCs range from 0.70 to 0.81; with small groups these differences are within noise.
- The simulated duration shift raises the mean predicted PD from 0.308 to 0.329 and gives a score PSI of 0.026, which is small by the usual rule of thumb.

Figures are in `figures/`; result tables are in `outputs/`.

## Limitations

- A small historical snapshot with no time dimension: real model deterioration and month-by-month monitoring cannot be assessed.
- Bad credits are over-sampled, so predicted PDs are not portfolio default rates.
- 300 test cases and 90 bad credits give wide intervals; segment results are uncertain, especially in small groups.
- One random split; repeated splits or nested cross-validation would give a steadier model comparison.
- The inputs include applicant sex and foreign-worker status, which a model used for real lending today could not use. They are kept because they are part of the public dataset.
- Model governance checks go well beyond what is shown here. This project is for learning and method comparison, not lending decisions.

## How to run

```
pip install -r requirements.txt
cd notebooks
jupyter notebook
```

Run A, B, C1, C2, C3, D, E in that order.
