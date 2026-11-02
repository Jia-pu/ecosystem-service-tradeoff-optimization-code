# Input Data Requirements

This repository is a code-only release. The modelling dataset, trained model binaries, and generated outputs are not included.

Supply the matched modelling CSV to `run.py --input PATH`. The required columns are:

```text
ID,row,col,x,y_coord,Temperature,DEM,Slope,GDP,POP,NL,PD,SHDI,CONTAG,Area_mn,GP-WY TOP
```

The full `model-validation` stage also runs a twelve-factor comparison and therefore additionally requires:

```text
NDVI,Precipitation
```

`NDVI` and `Precipitation` are comparison variables; they are not predictors in the final ten-factor model. If only the ten-factor table is available, use `--stage fit-fixed`. This fits the archived fixed configuration but does not rerun or replace random or spatial cross-validation.

The values must be the exact transformed, dimensionless feature coordinates used by the selected archived model version. Do not substitute raw-unit data or earlier workbooks without rebuilding and documenting the preprocessing workflow.

The data provider, licence, and any restrictions on redistribution should be documented by the data owner before any input file is made public.
