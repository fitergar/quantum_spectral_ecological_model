# Quantum Spectral Ecological Model — Chimalapas / Ptychohyla euthysanota

This repository accompanies the article: **"AI based quantum motivated spectral modeling of _Ptychohyla euthysanota_ in the Chimalapas montane forest"** and contains the full computational pipeline used to produce the results reported therein.

The purpose of this work is to infer **qualitative spatial structure** of a species distribution in a remote, data-scarce environment, where field observations are extremely sparse and spatially clustered. Rather than attempting precise abundance estimation, the framework focuses on recovering **ecologically credible spatial patterns**, such as strong stream affinity and fine-scale variation along river corridors.

The approach integrates three main ingredients:

1. a probabilistic formulation of species occupancy using Gibbs measures,
2. spectral analysis of discrete Schrödinger operators defined on spatial graphs, and
3. neural networks to extrapolate inferred habitat suitability across the landscape.

---

## 1. Methodological overview and relation to the code

This section summarizes the modeling workflow at a conceptual level and explains how each methodological step is reflected in the structure of the code. Detailed file-level descriptions and execution instructions are provided in Section 2.

---

### 1.1 Spatial discretization and GIS-based preprocessing

The study region is represented as a **two-dimensional spatial grid** in UTM coordinates using the reference system **EPSG:32615 (WGS84 / UTM zone 15N)**. Each grid cell corresponds to a square of fixed side length (10 m in the original study), chosen to match the positional uncertainty of handheld GPS measurements collected during field surveys.

All spatial preprocessing is carried out **externally**, prior to running the Python modeling pipeline. In the original study, this preprocessing was performed using **QGIS**, but the modeling code itself is agnostic to the specific GIS software used. Any workflow capable of producing equivalent tabular outputs can be substituted.

The GIS preprocessing stage produces the following spatial data products:

- a regular grid covering the study region,
- centroid coordinates for each grid cell,
- a vector representation of the river network,
- elevation data referenced to mean sea level,
- rasterized satellite imagery aligned to the grid.

Hydrological and topographic layers are obtained from **INEGI (Instituto Nacional de Estadística y Geografía)** open datasets. In particular, vector hydrography and topography correspond to the INEGI E15C55 La Gloria dataset (Serie III, 1:50 000), available at:

https://www.inegi.org.mx/app/biblioteca/ficha.html?upc=889463497363

Elevation data are derived from the INEGI **Continental Relief** products, available at:

https://www.inegi.org.mx/temas/relieve/continental/#descargas

Elevation rasters are reprojected to the project coordinate system (EPSG:32615) and sampled at grid centroids to obtain per-cell elevation values. Due to their large file size, the original elevation raster files are **not included in this repository**, but they are publicly available from INEGI and can be readily regenerated following the same procedure.

Satellite imagery is accessed through **QuickMapServices** within QGIS and rasterized onto the same grid used for spatial discretization. The resulting raster layers are aligned to the project coordinate system to ensure spatial consistency.

Most spatial products generated during this stage (GeoPackages, shapefiles, and selected raster-derived products) are stored in the repository under the `data/Gpx/` directory. These files are included to facilitate reproducibility, but users are free to regenerate them using other GIS software or alternative open datasets, provided the resulting grid-level attributes are equivalent.

---

### 1.2 River geometry, distance, and relative elevation

The river network plays a central role in the modeling framework, as _Ptychohyla euthysanota_ is strongly associated with stream environments. From the spatial grid and the river geometry, two key covariates are derived for each cell:

- **distance to the river**, defined as the distance from the grid cell centroid to the nearest river segment,
- **relative height with respect to the river**, capturing elevation differences between a cell and nearby river locations.

These quantities encode large-scale ecological constraints—particularly stream affinity and topographic position relative to the river—without imposing them directly in the probabilistic model.

In the original study, the river network was digitized in QGIS as a dedicated vector layer aligned to the project grid. The study region was defined by drawing a polygon enclosing the relevant section of the river corridor, and the grid was clipped to this region. River geometry was then converted into a set of reference cells or points aligned with the grid resolution, allowing distances from grid centroids to the river to be computed consistently.

Elevation values sampled at grid centroids were used to derive a relative elevation measure by comparing each cell's elevation to that of nearby river cells. This relative height is intended to capture local topographic structure along the river corridor, rather than absolute elevation above sea level.

It is important to emphasize that this GIS-based procedure is **not unique**. Distance to river and relative elevation can be computed using a variety of equivalent methods and tools, provided that the resulting quantities are defined consistently on the grid. The modeling framework does not depend on a specific implementation, only on the availability of these covariates at the grid-cell level.

Within the Python pipeline, the computation and attachment of distance-to-river and relative-elevation covariates are handled by the script `preprocess_grid_to_dataset.py`. This script takes as input a base grid CSV containing coordinates, elevation above sea level, and a river indicator, and augments it with these derived quantities in a form suitable for downstream modeling.

---

### 1.3 Satellite imagery and texture information

To capture fine-scale environmental variation along the river corridor, the framework incorporates information derived from satellite imagery. These data serve as proxies for local surface texture and habitat heterogeneity that are not captured by distance to river or elevation alone.

In the original study, satellite maps were accessed through **QuickMapServices** within QGIS and rasterized onto the same spatial grid used for discretization. Raster layers corresponding to the red, green, and blue channels were generated, reprojected to the project coordinate system (EPSG:32615), and pixelated so that each grid cell corresponded to a single raster value per channel.

RGB values were then sampled at grid centroids and exported to tabular form (CSV files). These CSVs were merged with the main grid dataset, ensuring that each cell carried a consistent set of satellite-derived covariates aligned with its spatial location.

Within the Python pipeline, these RGB values are attached to each grid cell by `preprocess_grid_to_dataset.py` and subsequently combined into greyscale descriptors. The greyscale quantities are used as inputs to the neural network that modulates habitat suitability locally along the river corridor.

As with the river-related covariates, this satellite-processing workflow is **not unique**. Any raster source capable of providing spatially aligned texture information can be used, and alternative GIS or remote-sensing tools may be substituted. QuickMapServices is used solely as a convenient interface for accessing open map tiles; the modeling framework itself does not depend on a specific satellite provider, only on the availability of grid-aligned raster values exported in tabular form.

---

### 1.4 Field observations, training region, and smoothing

Field observations consist of georeferenced detections of _Ptychohyla euthysanota_ collected during a limited number of expeditions. Due to the remoteness of the study area and the logistical constraints of access, detections are extremely sparse and concentrated almost exclusively near the river network. After discretization, the majority of grid cells contain zero observations.

To stabilize inference and ensure that the inverse spectral step is numerically well behaved, the analysis is restricted to a **training region** located along the surveyed river corridor. This region corresponds to the area where field observations were collected and where the spatial support of the data is sufficiently dense to allow reliable reconstruction of the habitat-suitability potential.

In the current implementation, the training region is defined explicitly in UTM coordinates (EPSG:32615) as the union of several axis-aligned rectangular subregions that follow the sampled river corridor. A grid cell with centroid coordinates $(X, Y)$ is included in the training region if it satisfies at least one of the following conditions:

- $327634.1 < X < 327945$ and $1856536.44 < Y < 1856805.7$
- $327727.0 < X < 327946$ and $1856805.7 \le Y < 1856881.0$
- $327805.0 < X < 327946$ and $1856881.0 \le Y < 1856903.0$
- $327807.0 < X < 327946$ and $1856903.0 \le Y < 1856923.0$

The training region is further restricted to cells within a fixed maximum distance from the river: $d_{\text{rio}} \le d_{\text{max}}$.

This explicit geometric definition is encoded directly in the preparation stage of the pipeline and can be modified to accommodate different study regions or river geometries. The use of simple coordinate inequalities avoids reliance on opaque spatial joins and makes the definition of the training domain fully transparent and reproducible.

Once the training region is selected, raw detection counts (NUMPOINTS) are spatially regularized. This step is implemented in `prepare.py` and consists of applying a **Gaussian smoothing** procedure on the grid. The result is a strictly positive, continuous proxy for relative occupancy, denoted by $\psi$, defined on the training region.

The smoothing step serves two purposes:

1. it regularizes the extremely sparse observation signal implied by the field data, and
2. it ensures strict positivity of $\psi$, which is required for the inverse reconstruction of the habitat-suitability potential.

Importantly, this procedure introduces no additional ecological assumptions beyond spatial continuity at the scale of the grid. It is a numerical regularization step that enables stable spectral inference while preserving the large-scale spatial structure implied by the observations.

The output of this stage is a canonical, run-specific dataset containing the smoothed occupancy proxy and all associated covariates, which is subsequently used for potential inference, neural network training, and prediction.

---

### 1.5 Probabilistic formulation and inverse spectral inference

Species occupancy on the spatial grid is modeled as a collection of real-valued variables defined on the grid cells, with local spatial dependence encoded through nearest-neighbor interactions. This induces a Markov random field structure in which only adjacent cells interact.

Under these locality assumptions, the joint distribution of the occupancy field can be written in Gibbs form,

$$P(x) \propto \exp(-\mathcal{H}(x))$$

where the Hamiltonian $\mathcal{H}$ represents a balance between spatial smoothness and local habitat suitability.

In the model used here, the Hamiltonian consists of two components: a dispersal term that penalizes differences between neighboring cells, and a site-dependent potential that encodes habitat suitability. Together, these define a quadratic energy that can be written in operator form as

$$\mathcal{H}(x) = \langle x,(L + \mathrm{diag}(V))x\rangle$$

where $L$ is the graph Laplacian of the grid and $V$ is the environmental potential. This representation makes the connection to spectral theory explicit. The operator

$$H = L + \mathrm{diag}(V)$$

is a discrete Schrödinger operator on the grid graph, and the most probable spatial configuration of the species corresponds to its ground state.

On the training region defined in Section 1.4, the smoothed occupancy proxy $\psi$ is strictly positive. Rather than prescribing a parametric form for habitat suitability, the potential is reconstructed directly by enforcing the ground-state condition

$$(L + \mathrm{diag}(V))\psi = 0$$

This yields a pointwise expression for the potential,

$$V = -(L\psi)/ \psi$$

which uniquely determines habitat suitability on the training region under mild connectivity assumptions.

In the codebase, this inverse-spectral reconstruction is implemented using sparse linear algebra routines. Graph Laplacians and related operators are defined in `linalg.py`, while feature handling and dataset assembly are managed by utilities in `featurize.py`.

The inferred potential on the training region serves as the target variable for the neural-network models described in Section 1.6. In this way, probabilistic modeling and spectral inference provide the bridge between sparse field observations and data-driven extrapolation across the landscape.

---
### 1.6 Learning and extrapolation of habitat suitability

The environmental potential reconstructed in Section 1.5 is defined only on the training region. To extrapolate habitat suitability across the full landscape, this potential is learned as a function of environmental covariates using neural networks.

The neural-network architectures used throughout the pipeline are defined in `models.py`. This module contains the model classes that implement the distance-based driver and its locally modulated extensions, ensuring consistent input dimensionality and parameterization across training and prediction stages.

Two models are trained in sequence:

- a **distance-to-river driver**, which captures the dominant large-scale dependence of habitat suitability on proximity to the river network and is trained by `train_driver.py`,
- a **greyscale- and relative-elevation-modulated model**, which refines the driver by incorporating satellite-derived texture and relative height information near the river corridor and is trained by `train_grey.py`.

The driver model is trained first to learn the global river-distance dependence. Its parameters are then frozen, and the modulated model is trained on top of it to capture fine-scale spatial variation along the river without altering the large-scale structure.

The output of this stage is a learned, spatially continuous representation of the environmental potential that can be evaluated at any grid cell within the prediction domain.

---

### 1.7 Local spectral prediction and spatial averaging

Predicted species distributions are generated locally along the river network using the learned potential. For each river-adjacent grid cell, a square neighborhood is constructed, defining an induced subgraph of the spatial grid.

Within each neighborhood, the learned potential is evaluated, the corresponding discrete Schrödinger operator is formed, and its ground state is computed numerically. The squared amplitude of the ground state, $|\psi|^2$, is interpreted as a local probability distribution over that neighborhood.

Because neighborhoods overlap spatially, each grid cell may receive multiple local probability estimates. These local predictions are combined by simple averaging across all neighborhoods containing the cell, yielding a stable global prediction that mitigates boundary effects, scaling ambiguities, and geometric artifacts.

These prediction steps are implemented in `predict_driver_local.py` and `predict_gray_local.py`. Final prediction tables and intermediate outputs are written to the corresponding run directory under `outputs/`, enabling direct comparison between distance-only and fully modulated models.

## 2. Reproducible run protocol

### 2.1 Requirements (software + environment)

This project is a pure Python pipeline (plus PyTorch) that reads and writes CSV files. It does **not** require QGIS at runtime: all GIS work happens before the Python stage. The Python code assumes you already have the grid-level CSV inputs in `data/`.

Below are the practical requirements to run the code exactly as intended.

---

#### Python version

Use **Python 3.10 or newer**.

The scripts rely on modern standard-library features (`pathlib`, typing, argparse patterns) and the stack is most stable on 3.10+.

---

#### Required Python packages

At minimum, you need:

- **numpy**
- **pandas**
- **scipy**
- **torch** (PyTorch)
- **numba** (used in `src/linalg.py` for the JIT-compiled iteration routine)

The project explicitly imports these across the pipeline:
- `torch` is required for training and for evaluating the models during prediction. 
- `scipy` is required for prediction (sparse inverse) and for smoothing in `prepare.py` (you already saw this earlier). 
- `numba` is required because `minevec_iter` is decorated with `@njit` in `src/linalg.py`. 

If `numba` is missing, the import of `src/linalg.py` will fail.

---

#### Optional but recommended

- **CUDA-capable GPU + CUDA-enabled PyTorch**  
  Training and prediction can run on CPU, but training for 100k epochs is noticeably faster with CUDA. Both `train_driver.py` and `train_grey.py` support a `--cuda` flag.   
  Prediction scripts also accept `--cuda` for model evaluation (the eigen solve itself still uses NumPy/SciPy). 

---

#### Installation suggestions (two common options)

**Option A: venv + pip**
1. Create and activate a virtual environment
2. Install packages

Example (Linux/macOS):
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install numpy pandas scipy numba torch
```
**Option B: conda**
```bash
conda create -n qsem python=3.10
conda activate qsem

conda install numpy pandas scipy numba
pip install torch
```

(Use the PyTorch install command appropriate for your system if you want GPU support.)
### 2.1 Requirements (clarification about citations in this README)

If you are editing this README using text generated by ChatGPT, you may occasionally see strange inline tokens that look like:

``

These are **internal tool citations** used by ChatGPT to track which uploaded file supported a statement. They are **not part of the project**, they do **not** belong in the README, and they can be safely removed.

Everything in this README should be normal Markdown: headings, bullet lists, code blocks, and (optionally) GitHub-rendered math.

---

### 2.2 `src/utils.py` (paths, run directories, and naming conventions)

This repository uses a strict convention for where datasets, run artifacts, and outputs are stored. Instead of duplicating file paths inside every script, all scripts rely on a single helper module: `src/utils.py`.

The purpose of `src/utils.py` is **consistency**:
- every script reads inputs from the same locations,
- every script writes outputs using the same naming rules,
- and every run is reproducible because the output tree is predictable.

#### What `dataset_paths(dataset)` does

The most important function in `src/utils.py` is `dataset_paths(dataset)`. It takes a dataset name (for example `test1`) and returns the canonical locations for:

- the dataset CSV produced by preprocessing:
  - `data/<dataset>.csv`
- the per-run output directory:
  - `outputs/<dataset>/`
- the prepared dataset produced by the preparation stage:
  - `outputs/<dataset>/prepared.csv`

Other scripts then build on top of these canonical paths to write:
- `outputs/<dataset>/args_*.json` (saved command-line arguments)
- `outputs/<dataset>/model_*.pt` and `model_*_meta.json` (trained models + metadata)
- `outputs/<dataset>/*metrics*.csv` (training/evaluation logs)
- `outputs/<dataset>/*pred*.csv` (predictions)

This is why most scripts only ask for `--dataset <NAME>`: once the dataset name is known, every input/output file is determined automatically.

#### Helper functions you will see across scripts

`src/utils.py` also defines small helper utilities used across the pipeline. The exact set may evolve, but they typically cover:

- printing consistent status messages (progress logging)
- creating directories safely (`mkdir` behavior)
- lightweight argument saving (write `args_*.json`)
- simple error handling helpers (consistent failure messages)

These helpers are intentionally simple. The main design goal is to avoid “magic”: the repository should be runnable from the command line with predictable filenames and minimal hidden state.

#### Practical consequence for reproducibility

Because all paths are derived from `--dataset`:
- two users running the same commands with the same dataset name should produce the same output file locations,
- runs do not overwrite each other unless they reuse the same dataset name,
- and the `outputs/<dataset>/` directory contains a complete record of what was run (args files, prepared.csv, models, predictions).

This convention is what makes it possible to reproduce the results end-to-end from a clean checkout of the repository.

### 2.3 `src/preprocess_grid_to_dataset.py` — build the modeling dataset (distance/relief + RGB subpixels)

This is the **first Python step** of the pipeline. It takes:

1) a **base grid CSV** produced by GIS (cell centers + elevation + river indicator), and  
2) three **RGB tile CSVs** (exported from raster sampling),

and writes a single dataset CSV to `data/<dataset>.csv` that the rest of the scripts consume.

At a high level it does two things:

- **River geometry features**
  - `drio`: distance-to-river (computed from the grid + river indicator)
  - `Zdrio`: elevation relative to the nearest river cell

- **Satellite/RGB features**
  - attaches **25 aligned RGB “subpixels”** around each grid cell:
    - `R1..R25`, `G1..G25`, `B1..B25`
  - (optional) also stores the sampled subpixel coordinates `Xsub1..Xsub25`, `Ysub1..Ysub25`

This script is intentionally “dumb but reproducible”: it does not require shapefiles or rasters at runtime.
Everything is done from CSVs so the full run can be reproduced without QGIS installed, as long as the CSV inputs exist.

---

#### Inputs

All file paths default to the `data/` directory:

- `Gridvaluesraw.csv`  
  A grid table with **one row per grid cell**. At minimum it must include:
  - `X`, `Y`: cell centroid coordinates in **EPSG:32615**
  - `Z`: elevation above mean sea level (meters)
  - `Esrio`: river indicator (river cells satisfy `Esrio != 0`)

- `R2GL.csv`, `G2GL.csv`, `B2GL.csv`  
  Tables with sampled raster values for each channel. Each one must include:
  - `X`, `Y`: sample coordinates
  - exactly **one value column** (the script infers which column is the pixel value)

These RGB CSVs are assumed to represent a **dense set of samples** (subpixels) around each grid cell,
so the script can pick 25 nearby samples per cell.

---

#### Outputs

- `data/<dataset>.csv`  
  This is the “raw dataset” used by `prepare.py` and everything after.

If `--save-args` is set, the script also writes:

- `data/args_preprocess_grid_to_dataset.json`  
  A snapshot of the full command line and resolved parameters for reproducibility.

---

#### What exactly gets computed?

##### 1) `drio` and `Zdrio`

A cell is considered a **river cell** if `Esrio != 0`.

For every cell, the script finds the nearest river cell using **L1 distance** in UTM coordinates:

- `drio = min_r ( |X - X_r| + |Y - Y_r| )` over river cells `r`

It also stores a river-relative elevation difference:

- `Zdrio = Z_cell - Z_nearest_river_cell`

For river cells, the nearest river cell is itself, so:

- `drio = 0` and `Zdrio = 0`

**Important note:** the implementation uses a brute-force computation of all cell-to-river distances.
That is fine for moderate grid sizes, but it is not optimized for very large domains.

##### 2) 25 aligned RGB “subpixels” per cell: `R1..R25`, `G1..G25`, `B1..B25`

Each of the R/G/B CSVs is treated as a “cloud” of sampled raster values.

For each sample point `(x, y)`:

1. The script applies an (x, y) shift:
   - `x <- x + x_offset`
   - `y <- y + y_offset`

   This compensates for systematic coordinate offsets that can appear when exporting raster samples.

2. The shifted point is assigned to the **nearest valid grid center** `(cx, cy)`.

3. The sample is kept only if it lies within a Chebyshev (L∞) radius around the center:
   - `max(|x - cx|, |y - cy|) <= radius`

4. Within each grid cell, samples are keyed by their *relative offset* `(dx, dy)` (rounded):
   - `(dx, dy) = (x - cx, y - cy)`
   - rounded using `key_precision` decimal digits

This rounding is the key detail: it lets the script intersect the three channels (R, G, B)
so that `Rk`, `Gk`, `Bk` correspond to the **same** subpixel location.

Finally, the script sorts candidate subpixels by L∞ distance (closest first) and takes
the first `n_per_cell` common subpixels (default 25), writing:

- `R1..R25`, `G1..G25`, `B1..B25`

If `--save-subcoords` is enabled, it also writes:

- `Xsub1..Xsub25`, `Ysub1..Ysub25`

These are the (shifted) subpixel coordinates corresponding to the chosen samples.

---

#### Command-line usage

Basic run (using defaults in `data/`):

```bash
python -m src.preprocess_grid_to_dataset \
  --dataset test1 \
  --save-args
```

