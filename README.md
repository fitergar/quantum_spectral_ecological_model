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

Elevation values sampled at grid centroids were used to derive a relative elevation measure by comparing each cell’s elevation to that of nearby river cells. This relative height is intended to capture local topographic structure along the river corridor, rather than absolute elevation above sea level.

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

Field observations consist of georeferenced detections of _Ptychohyla euthysanota_ collected during a limited number of expeditions. Due to the remoteness of the region and the logistical constraints of access, detections are extremely sparse and concentrated almost exclusively near the river network.

After discretization, most grid cells contain zero observations. To stabilize inference, the analysis is restricted to a **training region** near the river, and raw detection counts are spatially regularized. This stage is implemented in the script `prepare.py`.

Specifically, this step:

- defines the spatial extent of the training region (currently specified internally in the code),
- applies **Gaussian smoothing** to raw counts to produce a strictly positive proxy for relative occupancy, denoted ψ,
- constructs the canonical working dataset used by all subsequent training and prediction steps.

The smoothing procedure introduces no additional ecological assumptions; it regularizes the sparse signal implied by the spatial support of the data and enables stable spectral inference.

---

### 1.4 Field observations, training region, and smoothing

Field observations consist of georeferenced detections of _Ptychohyla euthysanota_ collected during a limited number of expeditions. Due to the remoteness of the study area and the logistical constraints of access, detections are extremely sparse and concentrated almost exclusively near the river network. After discretization, the majority of grid cells contain zero observations.

To stabilize inference and ensure that the inverse spectral problem is well posed, the analysis is restricted to a **training region** located along the surveyed river corridor. This region corresponds to the area where field observations were collected and where the spatial support of the data is sufficiently dense to allow reliable reconstruction of the habitat-suitability potential.

In practice, the training region is defined as the union of several contiguous rectangular subregions aligned with the river geometry, expressed directly in UTM coordinates (EPSG:32615). These subregions form a narrow, elongated domain that follows the river course and reflects the spatial extent of the sampling effort. In addition, only cells within a fixed maximum distance from the river are retained, ensuring that training is confined to the near-river environment where detections are meaningful.

Conceptually, the training region can be described as:
- a union of axis-aligned rectangles in the \((X, Y)\) plane that trace the river corridor,
- intersected with a distance-to-river constraint \(d_{\mathrm{rio}} \le d_{\max}\).

The exact numerical bounds of these rectangles are encoded directly in the preparation stage of the pipeline and can be modified to accommodate different study regions or river geometries. This explicit geometric definition avoids reliance on opaque spatial joins and makes the training domain fully reproducible.

Once the training region is selected, raw detection counts (`NUMPOINTS`) are spatially regularized. This step is implemented in the script `prepare.py` and consists of applying a **Gaussian smoothing** procedure on the grid. The result is a strictly positive, continuous proxy for relative occupancy, denoted \(\psi\), defined on the training region.

The smoothing step serves two purposes:
1. it regularizes the extremely sparse observation signal implied by the field data,
2. it ensures strict positivity of \(\psi\), which is required for the inverse spectral reconstruction of the habitat-suitability potential.

Importantly, this procedure introduces no additional ecological assumptions beyond spatial continuity at the scale of the grid. It is a numerical regularization step that enables stable spectral inference while preserving the large-scale spatial structure implied by the data.

The output of this stage is a canonical, run-specific dataset containing the smoothed occupancy proxy and all associated covariates, which is subsequently used for potential inference, neural network training, and prediction.

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

