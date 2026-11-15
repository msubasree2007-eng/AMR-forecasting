# Forecasting national antimicrobial resistance from ECDC EARS-Net data

Code and results for the manuscript "Forecasting national antimicrobial resistance: a strictly time
ordered benchmark against persistence in European surveillance 
data" (Subasree M).

- Archive (Zenodo): https://doi.org/10.5281/zenodo.24192332
- Repository: https://github.com/msubasree2007-eng/AMR-forecasting

## What this repository contains

Strict real-time evaluation of persistence, drift, damped drift, simple
exponential smoothing (SES), Elastic Net and gradient boosting for national
resistance forecasts (horizons 1–5 years), a pre-specified gate, a locked test
on 2022–2024, a consumption case study (H4), a genomic case study (H3) and
split-conformal prediction intervals (H5).

Splits: training 2005–2018, validation 2019–2021, locked test 2022–2024.
A model forecasting from origin O uses only outcomes with target year ≤ O.

## Folder structure

```
analysis and forecasting scripts/
  amr_core.py                 shared functions (data prep, models, evaluation)
  step2_eda.py                exploratory analysis
  step3_gate.py               gate on the original evaluation
  step3b_strict_origin.py     strict real-time re-evaluation (validation)
  step3c_robustness.py        robustness checks
  step3d_nearzero.py          sensitivity to the E. coli carbapenems series
  step4_consumption_test.py   H4: antibiotic consumption features
  step5_intervals.py          H5: conformal intervals (validation)
  step6_locked_test.py        locked test 2022–2024
results/
  LOCKED_TEST_DONE.txt            marker file written by step6
  locked_point_forecasts.csv      locked-test point forecasts
  locked_per_combo_h3.csv         locked-test per-combination results, h>=3
  locked_intervals.csv            locked-test prediction intervals
  step3b_validation_results.txt   output of step3b
  step6_locked_test_results.txt   output of step6
  intervals_validation_ses.csv    validation intervals (SES-centred)
  dryrun_*.csv                    dry-run outputs on validation data
  *.log, environment.txt          run logs and software versions
```

## Data

Data are not redistributed here. Download them from the original sources:

- ECDC Surveillance Atlas of Infectious Diseases (EARS-Net): indicator
  "R - resistant isolates, percentage".
- ESAC-Net antimicrobial consumption data.
- Genomic panel: NCBI Pathogen Detection isolate exports for K. pneumoniae (136,299 isolates; AMRFinderPlus gene calls, collection date, location, isolation source).

Put the ECDC file in the folder you run from (or pass its path as the first
argument):
python step3b_strict_origin.py ECDC_surveillance_data_Antimicrobial_resistance__1_.csv

step4_consumption_test.py takes the ESAC file as the second argument
(default: ESAC_Net_consumption_clean_v2.csv, the cleaned file used in the
paper).
## How to run

```
pip install -r requirements.txt
python "analysis and forecasting scripts/step3b_strict_origin.py"
```

Run the steps in numerical order. `step6_locked_test.py` reads the locked
2022–2024 years; do not run it again while developing models.

## Locked test: execution history

step6_locked_test.py was intended as a single final evaluation but was
executed more than once (on or about 31 Oct, 4 Nov and 9 Nov 2026). Google
Colab sessions are ephemeral, so the guard file LOCKED_TEST_DONE.txt did not
persist between sessions. The outputs of the runs compared were identical, no
model or setting was changed between runs, and the reported numbers come from
the files in `results/`.

## Integrity check

```
sha256sum "analysis and forecasting scripts/amr_core.py"
sha256sum "analysis and forecasting scripts/step6_locked_test.py"
```

Expected hashes begin with:
- amr_core.py: 229157d37b43
- step6_locked_test.py: 5ebb38e44084

## Scope note

H4 (consumption) was evaluated on the validation years 2019–2021 only and was
not repeated on the locked test.

# Environment

Python 3.10; scikit-learn 1.8.0; see requirements.txt and
results/environment.txt.

## Licence and citation

Code: [MIT, see LICENSE]. If you use this work, cite the
Zenodo archive above (see CITATION.cff).
