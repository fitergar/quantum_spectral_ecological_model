# Spatial Quantum–ML Model for River-Driven Species Distribution

This repository contains the code used to reproduce the results of a study on
spatial species distribution modeling near river systems, combining:

- Discrete spatial graphs (grid-based Laplacians)
- Schrödinger-type operators
- Neural-network–parameterized potentials
- Remote-sensing–derived grayscale covariates

The application shown here focuses on amphibian observations in a riverine
environment.

---

## Repository structure
.
├── data/ # Raw and processed input datasets
├── src/ # Source code (preprocessing, training, prediction)
├── outputs/ # Experiment outputs (models, metrics, predictions)
└── README.md


---

## Requirements

- Python ≥ 3.9
- NumPy
- Pandas
- SciPy
- PyTorch
- Numba
- Seaborn (optional, for diagnostics)

It is recommended to use a virtual environment.

---

## Basic workflow (high level)

1. **Preprocess grid data**
   ```bash
   python src/preprocess_grid_to_dataset.py
