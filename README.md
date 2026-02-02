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

