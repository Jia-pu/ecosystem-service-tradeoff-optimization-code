# Ecosystem Service Trade-Off Optimization: Core Research Code

This repository contains the core Python code used for the XGBoost–SHAP, partial dependence, accumulated local effects, and spatial-priority analyses in the ecosystem-service trade-off optimization workflow.

> **Code-only release.** The input modelling dataset, trained model binaries, generated result tables, and figure outputs are not included. Users must supply the corresponding data and have permission to use it.

## 1. Repository structure

| Directory / file | Description |
|---|---|
| `01_XGBoost_SHAP/` | XGBoost model training and validation, archived model parameters, SHAP contributions and interaction analysis |
| `02_1D_PDP_Priority/` | One-dimensional PDP/ICE and ALE calculations, candidate response-breakpoint analysis, and priority zoning |
| `03_2D_PDP_Priority/` | Two-dimensional PDP surfaces, joint-support assessment, priority zoning, and figure generation |
| `04_3D_PDP_Priority/` | Three-dimensional PDP surfaces, priority zoning, CONTAG-conditioned contour slices, and figure generation |
| `common/` | Shared PDP, support, zoning, and plotting functions |
| `data/` | Input-data requirements and field descriptions |
| `run.py` | Unified command-line entry point |
| `requirements.txt` | Python package versions used by this code package |
| `SOURCE_MAP.csv` | Mapping from packaged scripts to their archived source files and packaging treatment |
| `PACKAGE_CHECKS.json` | Recorded checks performed when the code package was assembled |
| `SHA256SUMS.csv` | SHA-256 checksums for the files in the original code package |

## 2. Methods covered

The scripts are organized around the following workflow:

1. XGBoost model training and random / spatial-block validation.
2. SHAP feature-contribution and pairwise interaction analysis.
3. One-dimensional PDP/ICE and ALE analysis.
4. Two-dimensional and three-dimensional PDP analysis.
5. Joint-observation support assessment.
6. Q75-based spatial-priority classification and map generation.
7. CONTAG-conditioned two-dimensional contour slices for interpreting the three-factor PDP results.

The ten model predictors are expected in this order:

`Temperature`, `DEM`, `Slope`, `GDP`, `POP`, `NL`, `PD`, `SHDI`, `CONTAG`, `Area_mn`.

The response field is `GP-WY TOP`.

## 3. Software environment

The archived package recommends Python 3.12. Create and activate a virtual environment, then install the pinned dependencies.

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the available command-line stages:

```bash
python run.py --help
```

## 4. Input data

No modelling data are distributed in this repository.

Provide the matched modelling CSV with the fields listed in [`data/README_inputs.md`](data/README_inputs.md). The values must use the same transformed, dimensionless feature coordinates as the archived model version; do not substitute raw physical-unit values or earlier spreadsheets without reproducing the preprocessing workflow.

The full `model-validation` stage also evaluates a twelve-factor comparison and therefore requires the `NDVI` and `Precipitation` columns in addition to the ten predictors. These two variables are used for the comparison and are not predictors in the final ten-factor model.

The `fit-fixed` stage accepts the ten-factor input table, fits the archived fixed parameter configuration, and does **not** generate new cross-validation results. It must not be described as a rerun of spatial cross-validation.

## 5. Run the analysis

Run commands from the repository root.

### Option A — full archived model-validation workflow

Use the matched input table that includes the additional `NDVI` and `Precipitation` comparison fields:

```bash
python run.py --stage model-validation --input "validation_inputs/modeling_data_12_factor_comparison.csv"
python run.py --stage shap
python run.py --stage 1d
python run.py --stage 2d
python run.py --stage 3d
```

### Option B — fit the archived fixed configuration

Use this route only when the available table contains the ten final predictors and the response field:

```bash
python run.py --stage fit-fixed --input "PATH/TO/modeling_data_10_predictors.csv"
python run.py --stage shap
python run.py --stage 1d
python run.py --stage 2d
python run.py --stage 3d
```

**Important:** Option B does not repeat model selection, random cross-validation, spatial-block cross-validation, or the factor-set sensitivity analyses.

### Generate figures

Install or locate Times New Roman on your system and provide its font-file path:

```bash
python run.py --stage plot-1d --font "C:/Windows/Fonts/times.ttf"
python run.py --stage plot-2d --font "C:/Windows/Fonts/times.ttf"
python run.py --stage plot-3d --font "C:/Windows/Fonts/times.ttf"
```

For the CONTAG-conditioned contour slices, first complete the three-dimensional calculation:

```bash
python run.py --stage slices --font "C:/Windows/Fonts/times.ttf"
python run.py --stage plot-3d-slices --font "C:/Windows/Fonts/times.ttf"
```

Generated tables and model files are written under `validation/`; figures are written under `replotted_figures/`. These generated outputs are excluded from version control by `.gitignore`.

## 6. Analysis definitions and interpretation notes

- The one-dimensional analysis retains seven factors: `GDP`, `PD`, `NL`, `POP`, `CONTAG`, `Area_mn`, and `SHDI`. The archived final figure layout displays five of them: `GDP`, `PD`, `POP`, `CONTAG`, and `SHDI`.
- The two-dimensional combinations are `GDP × PD`, `POP × NL`, `PD × SHDI`, `SHDI × CONTAG`, and `PD × CONTAG`.
- The three-dimensional combinations are `PD × SHDI × CONTAG` and `GDP × PD × CONTAG`.
- Joint support is assessed using 20 quantile bins for two-factor combinations and 6 quantile bins per variable for three-factor combinations. A joint bin must contain at least 30 observations to be treated as supported.
- Priority I requires all three Q75 conditions to be satisfied: high PDP response, positive SHAP contribution, and observed TOP. Priority II requires exactly two of the three conditions. The code uses `-1` for unsupported / unclassified observations, `0` for general areas, `1` for Priority II, and `2` for Priority I.
- The positive-SHAP component is the sum of positive contributions for the selected factors; it is not a SHAP interaction value.
- Historical `C` and `T` labels in the one-dimensional figure are exploratory response-change candidates. They should not be presented as independently validated ecological thresholds.
- Some archived two- and three-dimensional surface displays include sparsely supported combinations. Consult the support tables; the revised CONTAG-conditioned slices mask cells with insufficient joint support.
- Predictor axes are transformed, dimensionless coordinates. They are not original physical-unit thresholds.

## 7. Reproducibility and validation status

`PACKAGE_CHECKS.json` records the checks completed during packaging. It reports syntax checks, comparisons against archived outputs, and zoning consistency checks. It also explicitly records that a complete model-training rerun, the 100-repeat bootstrap rerun, and a full plotting rerun were **not** performed as part of packaging.

The package therefore provides the archived analysis code and recorded packaging checks; downloading it alone does not establish that the complete workflow has been independently rerun on a newly supplied dataset.

## 8. Data, licensing, and citation

- **Data:** Input data are not included. Users are responsible for obtaining and using data under the applicable terms and permissions.
- **License:** No open-source license is included in this repository yet. Unless the authors add a license, reuse and redistribution of the code are not automatically granted.
- **Citation:** Please cite the associated research article when its final bibliographic details become available. The manuscript is not assigned a final publication DOI in this code package.

## 9. Provenance

`SOURCE_MAP.csv` documents the relationship between packaged scripts and the archived source scripts, including where files were copied, extracted, reorganized, or made portable. The analysis formulas and archived training scripts are retained as documented there.
