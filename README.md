# Quantum Spectral Ecological Model — Chimalapas / Ptychohyla euthysanota

This repository accompanies the article:

**“AI based quantum motivated spectral modeling of _Ptychohyla euthysanota_ in the Chimalapas montane forest”**

and contains the full computational pipeline used to produce the results reported therein.

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

The study region is represented as a **two-dimensional spatial grid** in UTM coordinates using the reference system EPSG:32615 (WGS84 / UTM zone 15N). Each grid cell corresponds to a square of fixed side length (10 m in the original study), chosen to match the spatial uncertainty of handheld GPS measurements collected during field surveys.

All spatial preprocessing is carried out **externally**, prior to running the Python modeling pipeline. In the original study, this preprocessing was performed using **QGIS**, but the modeling code itself is agnostic to the specific GIS software used. Any workflow capable of producing equivalent tabular outputs can be substituted.

The GIS preprocessing stage produces the following spatial data products:

- a regular grid (retícula) covering the study region,
- centroid coordinates for each grid cell,
- a vector representation of the river network,
- elevation data referenced to mean sea level,
- rasterized satellite imagery aligned to the grid.

Hydrological and topographic layers are obtained from **INEGI** open datasets. Elevation rasters are reprojected to the project coordinate system and sampled at grid centroids to obtain per-cell elevation values. All vector and raster layers are maintained in the same UTM projection to ensure spatial consistency throughout the pipeline.

Most spatial products generated during this stage (GeoPackages, shapefiles, rasters) are stored in the repository under the `data/Gpx/` directory. These files are included to facilitate reproducibility, but users are free to regenerate them using other software or alternative open datasets, provided the resulting grid-level attributes are equivalent.

---

### 1.2 River geometry, distance, and relative elevation

The river network plays a central role in the modeling framework. From the grid and the river geometry, two key covariates are derived for each cell:

- **distance to the river**, defined as the distance from the cell centroid to the nearest river cell,
- **relative height with respect to the river**, capturing elevation differences between a cell and nearby river segments.

These quantities encode large-scale ecological constraints—particularly stream affinity—without imposing them directly in the probabilistic model.

The computation of river distance and relative elevation is handled in the first stage of the Python pipeline by the script `preprocess_grid_to_dataset.py`, which takes as input a base grid CSV containing coordinates, elevation above sea level, and a river indicator, and augments it with these derived covariates.

---

### 1.3 Satellite imagery and texture information

To capture fine-scale environmental variation along the river corridor, the framework incorporates information derived from satellite imagery.

Satellite maps are accessed via **QuickMapServices** within QGIS and rasterized onto the same grid used for spatial discretization. Raster bands corresponding to red, green, and blue channels are extracted, pixelated to match the grid resolution, and sampled at grid centroids. The resulting RGB values are exported to CSV files and later merged into the main dataset.

Within the Python pipeline, these RGB values are attached to each grid cell by `preprocess_grid_to_dataset.py` and subsequently combined into greyscale descriptors that serve as proxies for local texture and surface properties.

The modeling framework does not depend on a specific satellite provider; QuickMapServices is used solely as a convenient interface for accessing open map tiles. Any equivalent raster source can be substituted, provided the imagery is aligned to the grid and exported in tabular form.

---

### 1.4 Field observations, training region, and smoothing

Field observations consist of georeferenced detections of _Ptychohyla euthysanota_ collected during a limited number of expeditions. Due to the remoteness of the region and the logistical constraints of access, detections are extremely sparse and concentrated almost exclusively near the river network.

After discretization, most grid cells contain zero observations. To stabilize inference, the analysis is restricted to a **training region** near the river, and raw detection counts are spatially regularized. This stage is implemented in the script `prepare.py`.

Specifically, this step:

- defines the spatial extent of the training region (currently specified internally in the code),
- applies **Gaussian smoothing** to raw counts to produce a strictly positive proxy for relative occupancy, denoted ψ,
- constructs the canonical working dataset used by all subsequent training and prediction steps.

The smoothing procedure introduces no additional ecological assumptions; it regularizes the sparse signal implied by the spatial support of the data and enables stable spectral inference.

---

### 1.5 Probabilistic formulation and inverse spectral inference

Species occupancy on the grid is modeled as a collection of locally interacting random variables. Under standard locality assumptions, the joint distribution admits a Gibbs representation whose energy consists of:

- a **dispersal term**, penalizing sharp spatial variations and encoded by the graph Laplacian of the grid,
- an **environmental potential**, representing habitat suitability at each cell.

This energy defines a discrete Schrödinger operator \(H = L + \mathrm{diag}(V)\), where \(L\) is the combinatorial Laplacian. A central consequence of this formulation is that the most probable spatial configuration of the species corresponds to the **ground state** of \(H\).

Given a strictly positive smoothed occupancy proxy ψ on the training region, the environmental potential is reconstructed pointwise via the inverse spectral relation
\[
V_\ell = -\frac{(L\psi)_\ell}{\psi_\ell}.
\]
Under mild connectivity assumptions, this reconstruction is unique. The inferred potential provides a quantitative description of habitat suitability consistent with the observed spatial structure and serves as the training target for the neural networks.

Linear-algebra routines required for Laplacian construction and spectral computations are implemented in `linalg.py`, with supporting feature construction in `featurize.py`.

---

### 1.6 Learning and extrapolation of habitat suitability

The inferred potential is defined only on the training region. To extrapolate habitat suitability across the full landscape, it is learned as a function of environmental covariates using neural networks.

Two models are trained:

- a **distance-to-river driver**, which captures the dominant large-scale dependence on river proximity and is trained by `train_driver.py`,
- a **greyscale- and relative-elevation-modulated model**, which refines the driver using satellite-derived texture and elevation information near the river corridor and is trained by `train_grey.py`.

The driver model is trained first and then frozen while fitting the modulated model.

---

### 1.7 Local spectral prediction and spatial averaging

Predicted species distributions are generated locally along the river network. Overlapping neighborhoods are constructed around river-adjacent cells; within each neighborhood, the learned potential is evaluated, the local Schrödinger operator is formed, and its ground state is computed. Local probabilities \(|\psi|^2\) are then averaged across overlapping neighborhoods to produce a stable global prediction.

These steps are implemented in `predict_driver_local.py` and `predict_gray_local.py`. Final prediction tables are written to the corresponding run directory under `outputs/`.

---

Section 2 describes the repository structure and provides a detailed, reproducible execution protocol for running each script.

