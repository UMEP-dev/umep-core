# UMEP Core

## Installation

```bash
pip install umep
```

Or with uv:

```bash
uv add umep
```

## Troubleshooting

If you encounter DLL or import errors (common on Windows), run the diagnostic tool:

```bash
umep-doctor
```

### Common Issues

**OSGeo4W / QGIS Users**

Do NOT pip install into the OSGeo4W Python environment. The pre-installed GDAL binaries will conflict with rasterio's bundled DLLs, causing errors like:

```
ImportError: DLL load failed while importing _base: The specified procedure could not be found.
```

Instead, create a separate virtual environment:

```bash
uv venv --python 3.12
.venv\Scripts\activate   # Windows
uv pip install umep
```

**Conda Alternative**

If you prefer conda, use conda-forge for the geospatial dependencies:

```bash
conda create -n umep -c conda-forge python=3.12 rasterio geopandas pyproj shapely
conda activate umep
pip install umep
```

## Development Setup

- Install `uv` package manager (e.g. `pip install uv`).
- Clone repo.
- Run `uv sync` from the directory where `pyproject.toml` is located to install `.venv` and packages.
- Select `.venv` Python environment in your IDE.
- FYI: Recommended VS Code settings and extensions are included in the repo.

## Demo

See the demo notebook file at [/demo.py](/demo.py).

Also, a test with GBG data is found in [/solweig_gbg_test.py](/solweig_gbg_test.py)

The demo and the test uses the datasets included in the tests folder

## Original code

The code reproduced in the `umep` folder is adapted from the original GPLv3-licensed code by Fredrik Lindberg, Ting Sun, Sue Grimmond, Yihao Tang, Nils Wallenberg.

The original code has been modified to work without QGIS to facilitate Python workflows.

The original code can be found at: [UMEP-processing](https://github.com/UMEP-dev/UMEP-processing).

This modified code is licensed under the GNU General Public License v3.0.

See the LICENSE file for details.

Please give all credit for UMEP code to the original authors and cite accordingly.

© Copyright 2018 - 2020, Fredrik Lindberg, Ting Sun, Sue Grimmond, Yihao Tang, Nils Wallenberg.

Lindberg F, Grimmond CSB, Gabey A, Huang B, Kent CW, Sun T, Theeuwes N, Järvi L, Ward H, Capel- Timms I, Chang YY, Jonsson P, Krave N, Liu D, Meyer D, Olofson F, Tan JG, Wästberg D, Xue L, Zhang Z (2018) Urban Multi-scale Environmental Predictor (UMEP) - An integrated tool for city-based climate services. Environmental Modelling and Software.99, 70-87 https://doi.org/10.1016/j.envsoft.2017.09.020

## Demo Data

Two seprated demo dataset are included

### ATENS (vector data)

#### Tree Canopies

Copernicus

#### Trees

https://walkable.cityofathens.gr/home

#### Buildings

http://gis.cityofathens.gr/layers/athens_geonode_data:geonode:c40solarmap

### Gothenburg (raster data)

Standard dataset used in tutorials (https://umep-docs.readthedocs.io/en/latest/Tutorials.html)

### TODOs

- [ ] Is first idx divisor in sun on wall a bug?