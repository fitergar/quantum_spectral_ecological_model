# Quantum Spectral Ecological Model — Chimalapas / Ptychohyla euthysanota

This repository contains the computational implementation accompanying the article:

**“AI based quantum motivated spectral modeling of _Ptychohyla euthysanota_ in the Chimalapas montane forest”**

The code implements a qualitative species distribution modeling framework designed for **extremely sparse and spatially clustered field observations**, where standard statistical approaches are unreliable. The method combines:

- probabilistic modeling via Gibbs measures,
- spectral analysis of discrete Schrödinger operators on graphs, and
- neural networks to extrapolate inferred habitat suitability across a landscape.

The objective is **not quantitative abundance estimation**, but rather the recovery of **ecologically credible spatial structure** (e.g. stream affinity) under severe data limitations.

---

## 1. Conceptual overview and mapping to code

This section explains the full pipeline and explicitly maps each methodological component in the paper to the corresponding code.

---

### 1.1 External preprocessing (GIS) and required inputs

Before running any Python scripts, the study area is discretized into a **2D grid** (UTM) and exported as a base CSV. In the original study, this was done using **QGIS**, but any GIS software is acceptable as long as it produces the same required fields.

#### Base grid CSV (required)
A CSV representing the grid must include:
- `X`, `Y` — UTM coordinates of the cell center
- `Z` — elevation above mean sea level (from an elevation raster)
- `esrio` — indicator for river cells (1 if cell intersects the river, else 0)
- `NUMPOINTS` — raw field detections per cell (0 for most cells)

#### Satellite RGB CSVs (required for the greyscale model)
The pipeline can incorporate satellite texture by attaching rasterized RGB samples per cell. In the original study, RGB values were obtained by rasterizing satellite imagery onto the grid using **QuickMapServices** (QGIS) and exporting per-cell RGB values to CSV files (e.g., `B2.csv`, `G2.csv`, `R2.csv`).

> **Tools and data sources used in the study**
> - GIS preprocessing: QGIS (free/open source)
> - Hydrology/topography inputs: INEGI open datasets
> - Satellite imagery sampling: QuickMapServices (QGIS plugin), then exported to CSV
>
> Users of this repository are free to generate equivalent inputs using other software, as long as the resulting CSVs contain the expected columns.

---

### 1.2 `preprocess_grid_to_dataset.py`: enrich grid with river geometry and RGB covariates

Once the base grid CSV and the RGB CSV files exist, the first Python step is:

-src/preprocess_grid_to_dataset.py


This script takes the base grid (cells + `esrio` + `Z`) and produces the **model-ready dataset** by computing additional covariates:

1. **Distance to the river (`drio`)**
   - Identifies river cells using `esrio`.
   - Computes, for each cell, the distance to the nearest river cell.
   - Stores the result as `drio` (meters).

2. **Relative height to the river (relative elevation)**
   - Uses `Z` plus the river geometry (via `esrio`) to compute a relative-elevation-type covariate:
     height of the cell relative to a local river reference.
   - This is the “relative elevation with respect to the river” term described in the paper and is used by the greyscale-modulated model.

3. **Attach satellite RGB values**
   - Reads the RGB CSVs (e.g. `B2/G2/R2`) produced by rasterizing satellite imagery onto the grid.
   - Merges those values into the main dataset so each cell includes satellite-derived covariates.

**Output:**
- A single CSV (placed in `data/` or as specified by args) containing:
  - `X, Y, id, NUMPOINTS`
  - `Z` (elevation)
  - `drio` (distance to river)
  - relative height to the river (relative elevation covariate)
  - RGB covariates needed to derive greyscale features

This output is the **entry point** for the rest of the pipeline.

---

### 1.3 `prepare.py`: define training region and smooth sparse observations (construct ψ)

Field observations are extremely sparse and concentrated near the river network. To stabilize inference, we restrict inference to a **training region** close to the river and apply spatial smoothing.

This is handled by:

-src/prepare.py


This script performs:

1. **Training region selection**
   - Defines a spatial subset of the grid used for training (typically a near-river region).
   - This selection is currently **hard-coded** in the script and must be modified directly when changing study regions or buffers.

2. **Gaussian smoothing of field detections**
   - Raw counts (`NUMPOINTS`) are smoothed using a Gaussian kernel on the grid.
   - Produces a strictly positive occupancy proxy ψ used for inverse spectral inference.
   - This step does not introduce new ecological information; it regularizes the sparse signal implied by the data.

3. **Construction of the canonical working table**
   - Produces the run-specific dataset used by all training and prediction scripts.

**Output:**
- `outputs/<run_name>/prepared.csv`

---

### 1.4 Dispersal modeling: graph Laplacian on the grid

Spatial interactions are modeled via nearest-neighbor coupling on the grid:

- each cell is a vertex,
- edges connect cells sharing a side (4-neighborhood),
- dispersal is encoded via the combinatorial Laplacian \(L\).

**Code:**
- `src/linalg.py` — Laplacian construction and spectral solvers
- `src/featurize.py` — feature/tensor construction aligned with grid order

---

### 1.5 Inverse spectral inference of the habitat suitability potential V

Given a strictly positive smoothed occupancy proxy ψ on the training region, we infer a site-dependent potential:

\[
V_\ell = -\frac{(L\psi)_\ell}{\psi_\ell}.
\]

This defines a discrete Schrödinger operator:

\[
H = L + \mathrm{diag}(V),
\]

whose ground state matches ψ (up to normalization). Under mild connectivity assumptions this mapping from ψ to V is unique.

**Implementation notes:**
- The inferred potential is computed numerically as part of the training pipeline.
- Linear algebra utilities and stable operations live in `src/linalg.py`.

The inferred potential is the **training target** for the neural networks.

---

### 1.6 Learning the potential with neural networks

Two neural models are trained:

#### (a) Distance-to-river driver model
Learns a baseline potential depending only on `drio`.

**Script:**
-src/train_driver.py


#### (b) Greyscale/relative-height modulated model
Refines the driver using satellite texture (greyscale derived from RGB) and relative elevation with respect to the river, activated primarily near the river through a distance-based gate.

**Script:**
-src/train_grey.py


The driver is trained first and then frozen while training the modulated model.

---

### 1.7 Prediction via local spectral ground states and neighborhood averaging

Predictions are generated locally along the river corridor:

1. Construct overlapping square neighborhoods centered on near-river cells.
2. For each neighborhood:
   - evaluate the learned potential,
   - build \(H = L + \mathrm{diag}(V)\),
   - compute its ground state,
   - interpret probability as \(|\psi|^2\).
3. Average overlapping neighborhood probabilities to mitigate boundary/degree artefacts and scaling ambiguity.

**Scripts:**
-src/predict_driver_local.py
-src/predict_gray_local.py


**Outputs:**
- Final averaged prediction tables in `outputs/<run_name>/`.

---
