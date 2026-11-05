# Forecasting National Antimicrobial Resistance

## Project Overview

This project investigates whether machine learning, genomic features, and antimicrobial consumption data can improve the forecasting of national antimicrobial resistance (AMR) trends compared with conventional statistical baselines.

The study uses a strictly time-ordered forecasting framework to evaluate predictive performance and reduce the risk of temporal data leakage.

The primary analysis uses national antimicrobial resistance surveillance data from the **European Centre for Disease Prevention and Control (ECDC) – EARS-Net**.

## Research Objectives

1. Evaluate national AMR forecasting performance against persistence and other statistical baselines.
2. Compare machine-learning approaches, including Elastic Net and Gradient Boosting, with conventional forecasting methods.
3. Assess the robustness of results under alternative eligibility criteria.
4. Investigate whether antimicrobial consumption data provide additional predictive information.
5. Evaluate forecast uncertainty using prediction intervals.

## Data Sources

The project uses data from the following sources:

- **ECDC – EARS-Net:** National antimicrobial resistance surveillance data.
- **ESAC-Net:** Antimicrobial consumption surveillance data.
- **NCBI Pathogen Detection:** Genomic surveillance information used in the exploratory genomic analysis.

The raw datasets are not included in this repository. Users should obtain the data from the respective official sources and comply with their access and reuse conditions.

## Repository Structure

```text
AMR-forecasting/
│
├── analysis and forecasting scripts/
│   ├── amr_core.py
│   ├── step2_eda.py
│   ├── step3_gate.py
│   ├── step3b_strict_origin.py
│   ├── step3c_robustness.py
│   ├── step3d_nearzero.py
│   ├── step4_consumption_test.py
│   ├── step5_intervals.py
│   └── step6_locked_test.py
│
├── results/
│   ├── step3b_validation_results.txt
│   ├── step6_locked_test_results.txt
│   └── intervals_validation_ses.csv
│
└── README.md
```

## Analysis Workflow

| Script | Purpose |
|---|---|
| `amr_core.py` | Shared functions used by the analysis scripts. |
| `step2_eda.py` | Exploratory data analysis and initial data assessment. |
| `step3_gate.py` | Initial forecasting benchmark and validation gate. |
| `step3b_strict_origin.py` | Strict real-time validation with origin-specific training. |
| `step3c_robustness.py` | Robustness analysis. |
| `step3d_nearzero.py` | Sensitivity analysis for near-zero resistance values. |
| `step4_consumption_test.py` | Evaluation of antimicrobial consumption features. |
| `step5_intervals.py` | Forecast interval evaluation. |
| `step6_locked_test.py` | Final locked-test evaluation. |

## Temporal Evaluation Design

The forecasting framework separates the analysis into three periods:

| Period | Years | Purpose |
|---|---|---|
| Training | 2005–2018 | Model development and training |
| Validation | 2019–2021 | Model evaluation and validation |
| Locked test | 2022–2024 | Final out-of-sample evaluation |

The strict real-time validation procedure ensures that each forecast model is trained only using outcomes available at the corresponding forecast origin.

The locked-test evaluation is kept separate from the validation stage.

## Forecasting Methods

The benchmark includes:

- Persistence
- Drift
- Damped trend
- Simple Exponential Smoothing (SES)
- Elastic Net
- Gradient Boosting

Model performance is evaluated using forecast errors and relative error against the persistence baseline.

A relative error below 1 indicates improvement over persistence, while a value above 1 indicates worse performance.

## Main Findings

The validation analysis did not demonstrate consistent improvement from machine-learning models over conventional forecasting baselines.

Simple Exponential Smoothing remained competitive with persistence across the evaluated forecast horizons.

The findings highlight the importance of strict temporal validation and baseline comparisons when assessing national AMR forecasting models.

## Reproducibility

The Python scripts are provided to document the analysis workflow.

To reproduce the analysis:

1. Obtain the required surveillance datasets from their official sources.
2. Prepare the input files using the expected column structure and filenames.
3. Install the required Python dependencies.
4. Execute the scripts in the appropriate analysis order.
5. Keep validation and locked-test evaluations separate.

Some scripts depend on shared functions in `amr_core.py` and other scripts in the repository. Run them from the same project directory.

**Important:** The locked-test script is designed for a single final evaluation after the analysis pipeline has been frozen. Do not rerun it casually.

## Software

The analysis was developed in Python using Google Colab.

Core libraries include:

- NumPy
- pandas
- scikit-learn

Additional dependencies should be installed as required by individual scripts.

## Research Scope and Limitations

This project evaluates the forecastability of national AMR surveillance trends. It does not claim that machine learning universally outperforms conventional forecasting methods.

Results should be interpreted in the context of surveillance data availability, temporal coverage, country-level reporting differences, and the limitations of genomic and antimicrobial consumption data.

## Project Status

Research code and analysis outputs are being organized for reproducibility and academic review.

Further documentation, dependency specifications, and data preparation instructions may be updated as the project progresses.
