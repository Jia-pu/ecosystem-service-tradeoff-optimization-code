# Running Guide

This guide describes how to run the archived core analysis scripts from the repository root.

## 1. Repository modules

| Directory | Main contents |
|---|---|
| `01_XGBoost_SHAP/` | XGBoost training, training-fold parameter selection, random and spatial validation, predictor-set comparison, residual Moran's I, SHAP contributions, and archived parameters |
| `02_1D_PDP_Priority/` | Seven-factor one-dimensional PDP/ICE, ALE comparison, candidate response-breakpoint analysis, priority-zone calculation, five-factor figure layout, and an optional full 100-repeat bootstrap validation script |
| `03_2D_PDP_Priority/` | Five two-factor PDP combinations, joint-observation support, Q75 priority zoning, two-dimensional contours, and GIS figure layout |
| `04_3D_PDP_Priority/` | Two three-factor PDP combinations, joint-observation support, Q75 priority zoning, the archived 3D layout, and CONTAG-conditioned contour slices |
| `common/` | Shared analysis, support, priority-zoning, and plotting functions |
| `run.py` | Unified command-line entry point |

Do not remove `common/`; it contains functions shared by the analysis modules.

## 2. Environment and input

Python 3.12 is recommended. Create an isolated environment:

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The expected input fields are described in `data/README_inputs.md`. The final ten-predictor order is:

`Temperature`, `DEM`, `Slope`, `GDP`, `POP`, `NL`, `PD`, `SHDI`, `CONTAG`, `Area_mn`.

The response field is `GP-WY TOP`.

The scripts do not repeat the historical Gaussianization and multivariate reconstruction preprocessing. Input values must match the selected archived model version. These predictors are transformed, dimensionless feature values, not raw physical-unit measurements.

The default comparison input filename used by the scripts is:

`validation_inputs/modeling_data_12_factor_comparison.csv`

Place the permitted input file at that location, or supply a different path using `--input` where supported.

## 3. Run the workflow

Run commands from the repository root.

### Full archived model-validation workflow

This workflow requires the twelve-factor comparison table, including `NDVI` and `Precipitation`. These two variables are included for the predictor-set comparison; the final model uses only the ten predictors listed above.

```bash
python run.py --stage model-validation --input "validation_inputs/modeling_data_12_factor_comparison.csv"
python run.py --stage shap
python run.py --stage 1d
python run.py --stage 2d
python run.py --stage 3d
```

### Fit the archived fixed ten-factor configuration

If only the ten-predictor table is available, use:

```bash
python run.py --stage fit-fixed --input "PATH/TO/modeling_data_10_predictors.csv"
python run.py --stage shap
python run.py --stage 1d
python run.py --stage 2d
python run.py --stage 3d
```

Choose one of the two starting routes. Both routes are followed by the SHAP and 1D/2D/3D stages.

**Important:** `fit-fixed` refits the ten-factor model using the archived fixed parameters. It does not repeat hyperparameter selection, random cross-validation, spatial-block cross-validation, or the predictor-set comparison. Do not describe this route as a new spatial-validation run.

The one-dimensional stage also generates response arrays and SHAP outputs required by later stages. The 2D and 3D stages can be run independently when the required SHAP outputs already exist.

Outputs are written under `validation/data/` and `validation/models/`. The input is copied into the working directory; the source file is not modified.

## 4. Generate figures

Use a Times New Roman font installed on your system. On Windows, the default font path is usually:

`C:/Windows/Fonts/times.ttf`

Run:

```bash
python run.py --stage plot-1d --font "C:/Windows/Fonts/times.ttf"
python run.py --stage plot-2d --font "C:/Windows/Fonts/times.ttf"
python run.py --stage plot-3d --font "C:/Windows/Fonts/times.ttf"
```

### CONTAG-conditioned 3D contour slices

Complete the 3D calculation first, then generate the slices:

```bash
python run.py --stage slices --font "C:/Windows/Fonts/times.ttf"
python run.py --stage plot-3d-slices --font "C:/Windows/Fonts/times.ttf"
```

The slices fix CONTAG at its observed Q25, Q50, and Q75 values and calculate new model predictions. Unsupported portions are left blank. The revised 3D layout reads these slices.

JPG outputs are saved under `replotted_figures/`. Slice previews and arrays are saved under `validation/slices/`.

## 5. Scientific definitions and version notes

- This package preserves the archived code version associated with the 50-coordinate-kilometre spatial-block configuration and the final 1000-tree model. Validation statistics from other later code versions have not been mixed into this package.
- The spatial-block naming follows the archived scripts. The coordinate reference system and ground-distance/area conversions must be confirmed from the data documentation.
- The one-dimensional analysis retains seven predictors: `GDP`, `PD`, `NL`, `POP`, `CONTAG`, `Area_mn`, and `SHDI`. The final figure layout displays five: `GDP`, `PD`, `POP`, `CONTAG`, and `SHDI`.
- Two-dimensional combinations: `GDP × PD`, `POP × NL`, `PD × SHDI`, `SHDI × CONTAG`, and `PD × CONTAG`.
- Three-dimensional combinations: `PD × SHDI × CONTAG` and `GDP × PD × CONTAG`.
- Joint support uses 20 quantile bins for two-factor combinations and 6 quantile bins per variable for three-factor combinations. Each joint bin must contain at least 30 observations. Priority zoning is additionally constrained by the P1–P99 response-grid range.
- Priority I requires all three Q75 conditions to be met: high PDP response, positive SHAP contribution, and observed TOP. Priority II requires exactly two of the three conditions.
- The SHAP component is the sum of positive contributions for the selected predictors; it is not a SHAP interaction value. Zone codes are `-1` for unclassified, `0` for general, `1` for Priority II, and `2` for Priority I.
- The original GIS layout renders `-1` and `0` together as a light study-area footprint. Quantitative classification must use the separate codes in the CSV; unclassified cells must not be counted as general areas.
- The `C` values in the one-dimensional figure are historical exploratory markers. They must not be presented as newly detected stable thresholds. The `T` values are model-response candidates, not independently validated ecological thresholds.
- The original 2D and point-cloud layouts include predictions for some sparsely supported combinations. Support status is retained in the numerical tables. The revised embedded 3D slices leave unsupported regions blank.
- The plotting scripts were reorganized into portable scripts during the earlier packaging process. Pixel-for-pixel identity with historical JPG files is not guaranteed. The 1D redraw uses 81-point response arrays.
- This package-assembly task only organized, extracted, and checked the code. It did not rerun the complete spatial-validation workflow or the 100-repeat bootstrap. See `PACKAGE_CHECKS.json` for the recorded scope.
- `02_1D_PDP_Priority/optional_full_response_validation.py` is the optional full response-validation script. Run it from the repository root after completing the full model-validation workflow. It may take substantial time and updates response and bootstrap outputs.

## 6. Source mapping and reproducibility

`SOURCE_MAP.csv` records the source files and the extraction or modification applied to each packaged script. `SHA256SUMS.csv` records file checksums.

The training-validation script and optional full response-validation script retain their archived source code. Shared core functions were separated according to the original formulas, with an additional guard for cases with no supported observations and separate outputs by analysis dimension.

The 3D slice scripts were made portable by removing machine-specific paths and temporary dependencies and by using an explicit font entry point.

A public code repository should be accompanied by the permitted modelling data and preprocessing records if full reproduction is intended. A code-only release does not constitute public release of the underlying data.
